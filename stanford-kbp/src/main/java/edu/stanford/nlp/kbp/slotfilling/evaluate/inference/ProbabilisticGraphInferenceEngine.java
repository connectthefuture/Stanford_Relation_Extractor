package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.evaluate.EntityGraph;
import edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;
import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;

/**
 * Parent class for ProbabilisticGraphInference and BLNGraphInference
 */
public abstract class ProbabilisticGraphInferenceEngine extends GraphInferenceEngine {

  public static enum RulesMode {
    KBP_ONLY,
    REVERB_STRICT,
    REVERB
  }

  /**
   * The true known responses, for use in printing debug information
   */
  protected final GoldResponseSet goldResponses;

  protected final MLNText candidateRules = new MLNText();

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("ProbabilisticInference");

  /** Do not remove me -- this is the constructor invoked by reflection */
  @SuppressWarnings("UnusedDeclaration")
  public ProbabilisticGraphInferenceEngine() throws IOException {
    this(GoldResponseSet.empty(), Props.TEST_GRAPH_INFERENCE_RULES_FILES, Props.TEST_GRAPH_INFERENCE_RULES_CUTOFF,
            Props.TEST_GRAPH_INFERENCE_DEPTH);
  }

  /** Do not remove me -- this is the constructor invoked by reflection */
  @SuppressWarnings("UnusedDeclaration")
  public ProbabilisticGraphInferenceEngine(GoldResponseSet goldResponses) throws IOException {
    this(goldResponses, Props.TEST_GRAPH_INFERENCE_RULES_FILES, Props.TEST_GRAPH_INFERENCE_RULES_CUTOFF,
            Props.TEST_GRAPH_INFERENCE_DEPTH);
  }

  /** A more explicit constructor, passing a rules file and cutoff explicitly */
  public ProbabilisticGraphInferenceEngine(File rulesFile, double cutoff, int size_cutoff) {
    this(Collections.singletonList(rulesFile), cutoff, size_cutoff);
  }

  /** A more explicit constructor, passing a rules file and cutoff explicitly */
  public ProbabilisticGraphInferenceEngine(List<File> rulesFiles, double cutoff, int size_cutoff) {
    this(GoldResponseSet.empty(), rulesFiles, cutoff, size_cutoff);
  }

  protected ProbabilisticGraphInferenceEngine(GoldResponseSet goldResponses, List<File> rulesFiles, double cutoff, int size_cutoff) {
    this.goldResponses = goldResponses;

    startTrack("Loading candidate rules");
    candidateRules.predicates.addAll(kbpPredicates);
    for(File rulesFile : rulesFiles) {
      try {
        logf("Adding rules from %s", rulesFile.getPath() );
        BufferedReader rulesReader = IOUtils.getBufferedReaderFromClasspathOrFileSystem(rulesFile.getPath());
        candidateRules.mergeIn(MLNReader.parse(rulesReader));
        logf("Currently have %d rules", candidateRules.rules.size());

      } catch ( IOException ex ) {
        throw new RuntimeException(ex);
      }
    }

    // Filter out rules that don't apply
    {
      Iterator<MLNText.Rule> it = candidateRules.rules.listIterator();
      while(it.hasNext()) {
        MLNText.Rule rule = it.next();
        if( Math.abs(rule.weight) < cutoff ) it.remove();
        else if( rule.literals.size() > size_cutoff + 1) it.remove(); // +1 to account for the consequent
        else if(Props.TEST_GRAPH_INFERENCE_HACKS_NO_SPOUSE) {
          if(untypedRelation(rule.consequent().name).equals(RelationType.PER_SPOUSE.canonicalName) && rule.antecedents().size() > 0 ) {
            debug("Skipping rule " + rule + " because of spouse hack.");
            it.remove();
          }
        }
        else if(Props.TEST_GRAPH_INFERENCE_HACKS_NO_NEGATIVE_TRANSLATIONS) {
          if(rule.literals.size() == 2 && rule.weight < 0) {
            debug("Skipping rule " + rule + " because of negative translation hack.");
            it.remove();
          }
        }
      }
    }
    endTrack("Loading candidate rules");

  }

  static protected final Set<MLNText.Predicate> kbpPredicates = new HashSet<>();
  static {
    for (RelationType rel : RelationType.values()) {
      for(NERTag slotType : rel.validNamedEntityLabels) {
        kbpPredicates.add(new MLNText.Predicate(
                cleanRelation(rel.canonicalName, rel.entityType, slotType),
                rel.entityType.name.toUpperCase(),
                slotType.name.toUpperCase()
        )
        );
      }
    }
  }

  /**
   * Convert the edges of an entity graph into terms for an MLN.
   *
   *
   * @param graph The graph to construct the MLN from.
   * @param rules MLN rules that will be used.
   * @return An MLN for facts that can be changed, a list of assumed facts,
   *     the mapping from strings to the actual entities
   *
   * @see edu.stanford.nlp.kbp.slotfilling.evaluate.inference.MLNTextBuilder
   */
  public Pair<MLNText, Map<String, KBPEntity>> graphToMLN(EntityGraph graph, final MLNText rules, KBPEntity queryEntity) {
    // Create mapping to re-create types
    Map<String, KBPEntity> stringKBPEntityMap = new HashMap<>();

    Map<String, Pair<NERTag,NERTag>> openRelations = new HashMap<>();
    for( MLNText.Predicate in : rules.predicates ) {
      if(!in.closed)
        openRelations.put(in.name, Pair.makePair(NERTag.fromString(in.type1).get(), NERTag.fromString(in.type2).get()));
    }

    // Add evidence
    MLNText priors = new MLNText();
    for (KBPSlotFill fill : graph.getAllEdges()) {
      KBPEntity head = fill.key.getEntity();
      KBPEntity tail = fill.key.getSlotEntity().get();
      // Add to the map
      String arg1 = cleanEntity(head);
      String arg2 = cleanEntity(tail);
      stringKBPEntityMap.put(arg1, head);
      stringKBPEntityMap.put(arg2, tail);

      HashMap<String,Pair<NERTag, NERTag>> candidateNegativeRelations = new HashMap<>(openRelations);
      // Add positive evidence
      String relationName = cleanRelation(fill.key.relationName, fill.key.entityType, fill.key.slotType.getOrElse(NERTag.MISC));

      // Remove from the possible negative relations
      candidateNegativeRelations.remove(relationName);

      // Is this even a relation we care about?
      if(rules.getPredicateByName(relationName).isNothing()) continue;

      MLNText.Predicate pred = rules.getPredicateByName(relationName).get();
      MLNText.Literal literal = new MLNText.Literal(true, pred.name, arg1, arg2);
      priors.predicates.add(pred);

      // Is this a closed world relation in the graph?
      if( pred.closed ) {
        if(Props.TEST_GRAPH_INFERENCE_HACKS_SOFT_PRIORS) {
          priors.add( new MLNText.Rule(fill.score.getOrElse(Double.POSITIVE_INFINITY), literal) );
        } else {
          priors.add( new MLNText.Rule(Double.POSITIVE_INFINITY, literal) );
        }
      } else {
        priors.predicates.add(pred);
        // Translate scores to weighted rules
        if(fill.score.getOrElse(Double.POSITIVE_INFINITY).isInfinite())
          priors.add(new MLNText.Rule(Double.POSITIVE_INFINITY, literal));
        else {
          double prob = fill.score.getOrElse(0.);
          if(Props.TEST_GRAPH_INFERENCE_HACKS_ALWAYS_TRUE) {
            // HACK: Fix the priors to be 1.0 to be more like
            // SimpleGraphI.
            prob = 1.0;
          } else if(Props.TEST_GRAPH_INFERENCE_HACKS_SOFT_PRIORS) {
            // HACK: Rescale probability from .5 to 0.9 to keep
            // things fair and break ties.
            prob = 0.5 + 0.4 * prob;
          } else {
            // HACK: Rescale probability from .5 to 1.0 since it came out of the classifier
            prob = (1. + prob)/2.0;
          }
          double score = Math.log(prob / (1.0 - prob));
          priors.add(new MLNText.Rule(score, literal));
        }
      }
    }

    return Pair.makePair(priors, stringKBPEntityMap);
  }

  public List<MLNText.Literal> getQueryTerms(final MLNText rules, final KBPEntity entity) {
    HashSet<MLNText.Predicate> preds = new HashSet<>( CollectionUtils.filter(kbpPredicates, in -> (in.type1.equals(entity.type.name))));
    preds.retainAll(rules.predicates);

    return CollectionUtils.map(preds, in -> new MLNText.Literal(true, in.name, cleanEntity(entity), in.type2.toLowerCase() + "1"));
  }

  /**
   * Choose a subset of the rules that will only apply in this graph
   * @param graph - entity graph to extract rules from
   * @param entity - The subset of rules extracted are restricted to the set that might influence this entity
   * @return - A new rules set which is a subset of the original list.
   */
  public MLNText getRules(EntityGraph graph, final KBPEntity entity) {
    startTrack("Filtering rules for entity");
    // Create mapping from names to predicates
    HashMap<String, MLNText.Predicate> predicateNameMap = new HashMap<>();
    for(MLNText.Predicate pred : candidateRules.predicates) predicateNameMap.put(pred.name, pred);

    // Initialize the list of valid consequents and antecedents.
    final Set<String> validAntecedents =  new HashSet<>(CollectionUtils.map(graph.getAllEdges(), in -> cleanRelation(in)));
    if(Props.TEST_GRAPH_INFERENCE_RULES_MODE == RulesMode.KBP_ONLY) validAntecedents.retainAll(
            CollectionUtils.map(kbpPredicates, in -> in.name));
    final Set<String> validConsequents = new HashSet<>(CollectionUtils.filterMap(
            kbpPredicates,
        in -> {
          if (in.type1.equals(entity.type.name)) return Maybe.Just(in.name);
          else return Maybe.Nothing();
        }
    ));

    // Start moving rules from remainingRules to rules;
    final List<MLNText.Rule> remainingRules = new ArrayList<>(candidateRules.rules);
    final List<MLNText.Rule> rules = new ArrayList<>();
    int maxDepth = 10;
    int rulesSize = rules.size();

    // In a loop, add rules from consequent to antecedent
    for(int depth = 0; depth < maxDepth; depth++) {
      logf("(%d) %d rules (%d remain), with |V_c| = %d and |V_a| = %d", depth,
              rules.size(), remainingRules.size(),
              validConsequents.size(), validAntecedents.size());
      // Go through each rule
      ListIterator<MLNText.Rule> it = remainingRules.listIterator();
      while(it.hasNext()) {
        MLNText.Rule rule = it.next();
        // HACK: Just ignore all rules which output spouse
        if(Props.TEST_GRAPH_INFERENCE_HACKS_NO_SPOUSE) {
          if(untypedRelation(rule.consequent().name).equals(RelationType.PER_SPOUSE.canonicalName) && rule.antecedents().size() > 0 ) {
            debug("Skipping rule " + rule + " because of spouse hack.");
            continue;
          }
        }
        if(Props.TEST_GRAPH_INFERENCE_HACKS_NO_NEGATIVE_TRANSLATIONS) {
          if(rule.literals.size() == 2 && rule.weight < 0) {
            debug("Skipping rule " + rule + " because of negative translation hack.");
            continue;
          }
        }
        
        // If it's antecedents are contained within valid antecedent
        // and consequent is contained in the valid consequent, add to our list.
        if( CollectionUtils.all(rule.literals, in -> {
          if(!in.truth)
            return validAntecedents.contains(in.name);
          else
            return validConsequents.contains(in.name);
        })) {
          it.remove();
          rules.add(rule);
        }
      }

      // If we are in the "free-for-all version" than we update our valid antecedent and consequent list
      if(Props.TEST_GRAPH_INFERENCE_RULES_MODE == RulesMode.REVERB) {
        // If we've hit a fix point, break.
        if( rules.size() == rulesSize ) break;
        else rulesSize = rules.size();

        // Now update valid antecedents and consequents.
        validAntecedents.addAll(validConsequents);
        for(MLNText.Rule rule : rules) {
          for(MLNText.Literal literal : rule.literals) {
            if(!literal.truth)
              validConsequents.add(literal.name);
          }
        }
      } else
        break;
    }
    logf("# %d rules (%d remain), with |V_c| = %d and |V_a| = %d",
            rules.size(), remainingRules.size(),
            validConsequents.size(), validAntecedents.size());

    // Recompute valid antecedent and consequent based on the rules
    {
      validConsequents.clear();
      validAntecedents.clear();
      for(MLNText.Rule rule : rules)
        for(MLNText.Literal literal : rule.literals )
          if(literal.truth) validConsequents.add(literal.name);
          else validAntecedents.add(literal.name);
      // Add all the edges back in so that they are defined
      validAntecedents.addAll(CollectionUtils.map(graph.getAllEdges(), in -> cleanRelation(in)));
      // Add all KBP relations in so that we make inferences.
      validConsequents.addAll(CollectionUtils.filterMap(graph.getAllEdges(), in -> {
        if( in.key.getEntity().equals(entity) && in.key.hasKBPRelation()  )
          return Maybe.Just(cleanRelation(in));
        else return Maybe.Nothing();
      }));
    }

    MLNText filtered = new MLNText();
    {
      logf("# Creating %d rules with %d open and %d closed predicates.",
              rules.size(),
              validConsequents.size(), validAntecedents.size());

      // Now make all predicates in the antecedent but not in the consequent closed rules.
      for(String predicateName : validConsequents ) {
        if( predicateNameMap.get(predicateName) != null )
          filtered.predicates.add(predicateNameMap.get(predicateName).asOpen());
      }
      for(String predicateName : validAntecedents ) {
        if(!validConsequents.contains(predicateName)) {
          if( predicateNameMap.get(predicateName) != null )
            filtered.predicates.add(predicateNameMap.get(predicateName).asClosed());
        }
      }
      // Now just add all the rules!
      filtered.rules = rules;
    }
    endTrack("Filtering rules for entity");

    return filtered;
  }

  /**
   * Augment this entity graph with new relations, stemming from the pivot entity.
   * @param graph The graph to augment.
   * @param entity The entity to pivot on -- generally, the query entity.
   * @return The same graph as the input (this function is not functional!), but with potentially extra edges added.
   */
  @Override
  abstract public EntityGraph apply(EntityGraph graph, final KBPEntity entity);
}
