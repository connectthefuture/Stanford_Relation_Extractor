package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.EntityGraph;
import edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.math.SloppyMath;
import java.util.function.Function;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A very simple graph inference engine that matches antecedents, and if the rule
 * has high enough weight and all the antecedents match, it proposes the consequent.
 *
 * @author Gabor Angeli
 */
public class SimpleGraphInferenceEngine extends GraphInferenceEngine {
  Redwood.RedwoodChannels logger = Redwood.channels("RuleInfer");

  /**
   * An inference rule, represented as an antecedent and a consequent.
   */
  protected static class Rule {
    public final Set<KBTriple> antecedents;
    public final KBTriple consequent;
    public final double probability;

    private Rule(Set<KBTriple> antecedents, KBTriple consequent, double probability) {
      this.antecedents = antecedents;
      this.consequent = consequent;
      this.probability = probability;
    }
    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Rule)) return false;
      Rule rule = (Rule) o;
      return !(antecedents != null ? !antecedents.equals(rule.antecedents) : rule.antecedents != null) && !(consequent != null ? !consequent.equals(rule.consequent) : rule.consequent != null);
    }
    @Override
    public int hashCode() {
      int result = antecedents != null ? antecedents.hashCode() : 0;
      result = 31 * result + (consequent != null ? consequent.hashCode() : 0);
      return result;
    }
    @Override
    public String toString() {
      return StringUtils.join(antecedents, " ^ ") + " -> " + consequent;
    }

    public BoundRule bindConsequent(KBPair boundConsequent) {
      return new BoundRule(this, boundConsequent);
    }
  }

  /**
   * A (potentially partially) bound rule -- that is, a rule with some
   * variables already bound to concrete entities.
   */
  protected static class BoundRule extends Rule {
    private final Map<String, KBPEntity> bindings;

    private BoundRule(Rule rule, KBPair consequent) {
      super(rule.antecedents, rule.consequent, rule.probability);
      Map<String, KBPEntity> bindings = new HashMap<>();
      bindings.put(rule.consequent.entityName, consequent.getEntity());
      bindings.put(rule.consequent.slotValue, consequent.getSlotEntity().orCrash());
      this.bindings = Collections.unmodifiableMap(bindings);
    }
    private BoundRule(Rule rule, Map<String, KBPEntity> bindings, String var, KBPEntity binding) {
      super(rule.antecedents, rule.consequent, rule.probability);
      Map<String, KBPEntity> newBindings = new HashMap<>();
      newBindings.putAll(bindings);
      newBindings.put(var, binding);
      this.bindings = Collections.unmodifiableMap(newBindings);
    }

    /**
     * Returns the set of variables in this rule which have not been bound to a concrete entity yet.
     */
    public Set<String> freeVariables() {
      Set<String> freeVariables = new HashSet<>();
      for (KBTriple clause : antecedents) {
        if (!bindings.containsKey(clause.entityName)) { freeVariables.add(clause.entityName); }
        if (!bindings.containsKey(clause.slotValue)) { freeVariables.add(clause.slotValue); }
      }
      return freeVariables;
    }

    /**
     * Returns whether this candidate binding is consistent with the graph, given the other bindings so
     * far.
     *
     * @param graph The graph to check consistency over.
     * @param var The variable name to bind.
     * @param binding The candidate entity to check if this binding is valid for.
     * @return True if this is a consistent binding of variables.
     */
    public boolean isConsistent(EntityGraph graph, String var, KBPEntity binding) {
      // Compute
      for (KBTriple antecedent : antecedents) {
        if (antecedent.entityName.equals(var)) {
          // Case: found a clause where the entity is being bound
          if (antecedent.entityType != binding.type) { return false; }
          boolean knowSlotValue = bindings.containsKey(antecedent.slotValue);
          KBPEntity slotValueOrNull = bindings.get(antecedent.slotValue);
          boolean foundMatch = false;
          for (KBPSlotFill outgoingEdge : graph.outgoingEdgeIterable(binding)) {  // one of these edges must match
            if (outgoingEdge.key.relationName.equals(antecedent.relationName) &&  // the relation must match
                (!knowSlotValue || outgoingEdge.key.getSlotEntity().equalsOrElse(slotValueOrNull, false))) {  // the slot value must match, if already bound
              foundMatch = true;
              break;
            }
          }
          if (!foundMatch) {
            return false;
          }
        }
        if (antecedent.slotValue.equals(var)) {
          // Case: found a clause where the slot value is being bound
          if (!antecedent.slotType.equalsOrElse(binding.type, false)) { return false; }
          boolean knownEntity = bindings.containsKey(antecedent.entityName);
          KBPEntity entityOrNull = bindings.get(antecedent.entityName);
          boolean foundMatch = false;
          for (KBPSlotFill incomingEdge : graph.incomingEdgeIterable(binding)) {  // one of these edges must match
            if (incomingEdge.key.relationName.equals(antecedent.relationName) &&  // the relation must match
                (!knownEntity || incomingEdge.key.getEntity().equals(entityOrNull))) {  // the entity must match, if already bound
              foundMatch = true;
              break;
            }
          }
          if (!foundMatch) {
            return false;
          }
        }
      }
      return true;
    }

    /**
     * Bind the given variable to a concrete entity, returning a new rule instance.
     * @param var The variable to bind.
     * @param binding The entity to bind the variable to.
     * @return A new rule, with the given binding applied.
     */
    public BoundRule bind(String var, KBPEntity binding) {
      return new BoundRule(this, bindings, var, binding);
    }
  }

  /**
   * A mapping from the relation implied by the consequent, to a set of rules which would justify this relation.
   */
  protected Map<RelationType, Set<Rule>> antecedentsForRelation = new HashMap<>();
  /**
   * The true known responses, for use in printing debug information
   */
  private final GoldResponseSet goldResponses;
  /**
   * The minimum probability to propose new slot fills at
   */
  private final double minProb;

  /** Do not remove me -- this is the constructor invoked by reflection */
  @SuppressWarnings("UnusedDeclaration")
  public SimpleGraphInferenceEngine() throws IOException {
    this(Props.TEST_GRAPH_INFERENCE_RULES_FILES, Props.TEST_GRAPH_INFERENCE_RULES_CUTOFF);
  }

  /** Do not remove me -- this is the constructor invoked by reflection */
  @SuppressWarnings("UnusedDeclaration")
  public SimpleGraphInferenceEngine(GoldResponseSet goldResponses) throws IOException {
    this(GoldResponseSet.empty(), CollectionUtils.lazyMap(Props.TEST_GRAPH_INFERENCE_RULES_FILES, in -> {
      try {
        return IOUtils.getBufferedReaderFromClasspathOrFileSystem(in.getPath());
      } catch (IOException e) {
        throw new RuntimeIOException(e);
      }
    }), Props.TEST_GRAPH_INFERENCE_RULES_CUTOFF);
  }

  /** A more explicit constructor, passing a rules file and cutoff explicitly */
  public SimpleGraphInferenceEngine(BufferedReader rulesFile, double cutoff) {
    this(GoldResponseSet.empty(), Collections.singletonList(rulesFile), cutoff);
  }

  /** A more explicit constructor, passing a rules file and cutoff explicitly */
  public SimpleGraphInferenceEngine(List<File> rulesFiles, double cutoff) {
    this(GoldResponseSet.empty(), CollectionUtils.lazyMap(rulesFiles, in -> {
      try {
        return IOUtils.getBufferedReaderFromClasspathOrFileSystem(in.getPath());
      } catch (IOException e) {
        throw new RuntimeIOException(e);
      }
    }), cutoff);
  }

  /** The actual constructor implementation */
  private SimpleGraphInferenceEngine(GoldResponseSet goldResponses, List<BufferedReader> rulesFiles, double cutoff) {
    this.goldResponses = goldResponses;
    this.minProb = SloppyMath.sigmoid(cutoff);
    Pattern CLAUSE_REGEXP = Pattern.compile("!?([^\\(]+)\\(\\s*([^,]+)\\s*,\\s*([^\\)]+)\\s*\\)");
    for (RelationType rel : RelationType.values()) {
      antecedentsForRelation.put(rel, new HashSet<Rule>());
    }
    Map<String, Pair<NERTag, NERTag>> typeSignatures = new HashMap<>();
    // TODO(arun): Remove after ACL and fix this stupid rules reader.
    rulesFiles = Collections.singletonList(rulesFiles.get(rulesFiles.size()-1)); // The relevant rules file is always the last one.
    for(BufferedReader rulesReader : rulesFiles) {
      try {
        String line;
        while ( (line = rulesReader.readLine()) != null ) {
          if (line.contains("  ")) {
            // Variables for creating the rule
            Set<KBTriple> antecedents = new HashSet<>();
            KBTriple consequent = null;
            // Split the line into [score, rules]
            String[] elems = line.split("  ");
            double score = Double.parseDouble(elems[0]);
            if (score < cutoff) { continue; }
            // Get the clauses
            String[] clauses = elems[1].replaceAll("\\)v", ")  v  ").split("\\s+v\\s+");
            if (clauses.length - 1 > Props.TEST_GRAPH_INFERENCE_DEPTH) { continue; }
            for (String clause : clauses) {
              // For each clause, get the KBTriple expressed
              clause = clause.trim();
              Matcher matcher = CLAUSE_REGEXP.matcher(clause);
              if (!matcher.find()) {
                System.err.println(clause);
                throw new IllegalArgumentException("Invalid line of rule file (didn't match regexp): " + line);
              }
              Pair<NERTag, NERTag> signature = typeSignatures.get(matcher.group(1).trim());
              KBTriple triple = KBPNew.entName(matcher.group(2)).entType(signature.first)
                  .slotValue(matcher.group(3)).slotType(signature.second)
                  .rel(untypedRelation(matcher.group(1))).KBTriple();
              // Add this KBTriple to the appropriate part of the rule
              if (clause.startsWith("!")) {
                antecedents.add(triple);
              } else {
                if (consequent != null) {
                  throw new IllegalArgumentException("Invalid line of rule file (multiple consequents: " + line);
                }
                consequent = triple;
              }
            }
            // Add this rule, if appropriate
            if (consequent == null) {
              throw new IllegalArgumentException("Invalid line of rule file (no consequent): " + line);
            }
            for (RelationType rel : consequent.tryKbpRelation()) {
              antecedentsForRelation.get(rel).add(new Rule(antecedents, consequent, SloppyMath.sigmoid(score)));
            }
          } else {
            Matcher matcher = CLAUSE_REGEXP.matcher(line);
            if (!matcher.find()) {
              throw new IllegalArgumentException("Invalid type signature line: " + line);
            }
            if (!typeSignatures.containsKey(matcher.group(1).trim())) {
              typeSignatures.put(matcher.group(1).trim(),
                  Pair.makePair(
                      NERTag.fromString(matcher.group(2).trim()).orCrash(),
                      NERTag.fromString(matcher.group(3).trim()).orCrash()));
            }
          }
        }
      } catch (IOException e) {
        throw new RuntimeIOException(e);
      }
    }
  }

  /**
   * Check if this rule is supported by the entity graph, given that the consequent is
   * bound to a specific KBTriple.
   *
   * @param graph The entity graph to look for antecedent matches in.
   * @param rule The rule to match. The consequent of this rule is bound to the boundConsequent variable.
   * @param boundConsequent The actual entities in the graph that the consequent is bound to.
   * @return True if antecedents exist in this graph for the given rule.
   */
  protected boolean matches(EntityGraph graph, Rule rule, KBPair boundConsequent) {
    BoundRule boundRule = rule.bindConsequent(boundConsequent);
    return boundRule.isConsistent(graph, rule.consequent.entityName, boundConsequent.getEntity())
        && boundRule.isConsistent(graph, rule.consequent.slotValue, boundConsequent.getSlotEntity().orCrash())
        && !consistentBindings(graph, rule.bindConsequent(boundConsequent), true).isEmpty();
  }

  /**
   * Find all consistent bindings of variables to entities in the entity graph, given a
   * partially bound rule to start out from.
   *
   * @param graph The entity graph to find bindings in.
   * @param target A partially bound rule (e.g., usually bound with the candidate consequent) to find complete bindings from.
   * @param limit1Result If true, stop searching for consistent bindings after the first correct result.
   * @return A list of completely bound rules which are consistent with the entity graph.
   */
  protected List<BoundRule> consistentBindings(EntityGraph graph, final BoundRule target, boolean limit1Result) {
    // Base case: all variables are already bound
    if (target.freeVariables().isEmpty()) {
      return new ArrayList<BoundRule>(){{ add(target); }};
    }
    // Recursive case: try to search for a consistent binding
    List<BoundRule> rtn = new ArrayList<>();
    for (String var : target.freeVariables()) {
      for (KBPEntity candidateBinding : graph.getAllVertices()) {
        if (target.isConsistent(graph, var, candidateBinding)) {
          rtn.addAll(consistentBindings(graph, target.bind(var, candidateBinding), limit1Result));
          if (limit1Result && rtn.size() > 0) { return rtn; }
        }
      }
    }
    return rtn;
  }

  protected Maybe<KBPRelationProvenance> tryFindProvenance(EntityGraph graph, KBPEntity entity, KBPEntity slot, SimpleGraphInferenceEngine.Rule rule) {
    Maybe<KBPRelationProvenance> provenance = Maybe.Nothing();
    if (rule.antecedents.size() == 1) {  // If it's a translation rule...
      KBTriple antecedent = rule.antecedents.iterator().next();
      for (KBPSlotFill edge : graph.getOutgoingEdges(entity)) {
        if (edge.key.relationName.equals(antecedent.relationName) &&
            edge.key.slotType.equalsOrElse(slot.type, false) && edge.key.slotValue.equals(slot.name)) {
          if (edge.provenance.isDefined() && edge.provenance.get().isOfficial()) {
            provenance = Maybe.Just(edge.provenance.get());  // always take official provenances
          }
          provenance = provenance.orElse(edge.provenance);  // use that as provenance

        }
      }
    }
    return provenance;
  }


  /** {@inheritDoc} */
  @Override
  public EntityGraph apply(EntityGraph graph, KBPEntity entity) {
    // Print out raw reverb extractions from the query entity
    startTrack("Reverb Extractions");
    for (KBPSlotFill edge : graph.getOutgoingEdges(entity)) {
      if (!edge.key.hasKBPRelation()) {
        logger.debug(edge);
      }
    }
    endTrack("Reverb Extractions");

    // Do inference
    List<KBPSlotFill> toAdd = new ArrayList<>();
    for (KBPEntity sink : graph.getAllVertices()) {
      if (entity == sink) { continue; }
      for (RelationType candidate : RelationType.possibleRelationsBetween(entity.type, sink.type)) {
        for (Rule rule : antecedentsForRelation.get(candidate)) {
          if (rule.consequent.entityType != entity.type || !rule.consequent.slotType.equalsOrElse(sink.type, false)) { continue; }
          KBTriple candidateTriple = KBPNew.from(entity).slotValue(sink).rel(candidate).KBTriple();
          if (matches(graph, rule, candidateTriple)) {
            if (rule.probability > minProb) {
              String prefix = "?";
              if (this.goldResponses.isTrue(candidateTriple)) { prefix = "✔"; }
              else if (this.goldResponses.isFalse(candidateTriple)) { prefix = "✘"; }
              KBPSlotFill fill = KBPNew.from(entity).slotValue(sink).rel(candidate).score(0.5 + (rule.probability / 2.0))
                  .provenance(tryFindProvenance(graph, entity, sink, rule)).KBPSlotFill();
              if (!graph.getEdges(entity, sink).contains(fill) && !entity.equals(sink)) {  // If it's not already in the graph...
                logger.log("[" + prefix + "] inferred " + entity.name + " | " + candidate.canonicalName +
                    " [" + new DecimalFormat("0.000").format(rule.probability) + "] | " + sink.name +
                    "   {" + rule + "}");
                toAdd.add(fill);
              }
              break;
            }
          }
        }
      }
    }

    // Add inferred slots
    for (KBPSlotFill fill : toAdd) { graph.add(fill); }
    return graph;
  }
}
