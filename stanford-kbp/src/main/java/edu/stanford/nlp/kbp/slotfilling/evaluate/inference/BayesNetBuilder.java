package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.math.SloppyMath;
import edu.stanford.nlp.util.*;

import java.util.*;
import java.util.function.Function;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * An interface for constructing a Bayes Net,
 * as per {@link BayesNet}.
 *
 * @author Gabor Angeli
 */
public class BayesNetBuilder {

  private static final double logHalf = Math.log(0.5);

  public Map<String, Set<String>> validPairings = new HashMap<>();

  public enum FactorMergeMethod {
    NOISY_OR,
    HYBRID_OR,
    GENTLE_OR,
    GEOMETRIC_MEAN
  }

  static double clipLogProb(String name, double logProb) {
    if(logProb > Math.log(1.0 - 1e-4)) {
      debug("Rule " + name + " has too high of a logProb, " + logProb);
      logProb = Math.log(1.0 - 1e-4);
    } else if(logProb < Math.log(1e-4)) {
      debug("Rule " + name + " has too low of a logProb, " + logProb);
      logProb = Math.log(1e-4);
    }

    return logProb;
  }

  /**
   * Implements the factor between antecedents a_1, ..., a_n and consequent c
   * such that p(c | a_1, ..., a_n ) = p_0, and 1/2 otherwise.
   */
  protected static class EntailmentFactor implements BayesNet.Factor {
    public final String name;
    public final int[] antecedents;
    public final int consequent;
    public final double logProbTrue;
    public final double logProbFalse;

    @Override
    public String getName() { return name; }

    private EntailmentFactor(String name, double logProbTrue, double logProbFalse, int consequent, int... antecedents ) {
      this.name = name;
      this.antecedents = antecedents;
      this.consequent = consequent;
      this.logProbTrue = clipLogProb(name, logProbTrue);
      this.logProbFalse = clipLogProb(name, logProbFalse);
    }

    @Override
    public double logProb(boolean[] assignment) {
      boolean truth = true;
      for(int i = 0; i < antecedents.length && truth; i++)
        truth = assignment[antecedents[i]]; // as soon as this is false, we break
      if( truth ) {
        return assignment[consequent] ? logProbTrue : logProbFalse;
      } else {
        return logHalf;
      }
    }
    @Override
    public Collection<Integer> components() {
      List<Integer> components = new ArrayList<>(antecedents.length);
      for(int elem : antecedents) components.add(elem);
      components.add(consequent);
      return components;
    }

    public String toString(){
      return (name != null) ? name : super.toString();
    }
  }

  protected static class TableFactor implements BayesNet.Factor {
    final String name;

    final List<GroundedRule> rules;
    final GroundedRule prior;

    final int[] antecedents;
    final int consequent;

    final Map<Integer,Integer> antecedentMap;
    final Collection<Integer> components;
    final Map<String,Double> memoized = new HashMap<>();

    public TableFactor( TableFactor factor ) {
      name = factor.name;
      rules = factor.rules;
      prior = factor.prior;
      antecedents = factor.antecedents;
      consequent = factor.consequent;

      antecedentMap = factor.antecedentMap;
      components = factor.components;
    }

    public TableFactor( List<GroundedRule> rules ) {
      // Set up everything;
      cleanPriors(rules);
      GroundedRule prior = rules.get(0);
      assert rules.size() > 0;
      // Sort the rules
      Collections.sort(rules, (o1, o2) -> o2.antecedents.length - o1.antecedents.length);
      assert rules.get(0).antecedents.length >= rules.get(rules.size()-1).antecedents.length;

      String name;
      {
        StringBuilder sb = new StringBuilder();
        for(GroundedRule rule : rules)
          sb.append(rule.name).append("\n");
        name = sb.toString().trim();
      }

      // Get the consequents and antecedents.

      int consequent = rules.iterator().next().consequent;
      int[] antecedents;
      Map<Integer,Integer> antecedentMap = new HashMap<>();
      {
        Set<Integer> antecedentSet = new HashSet<Integer>();
        // Now collect all the antecedents to make the table.
        for(GroundedRule rule : rules) {
          assert consequent == rule.consequent;
          for(int antecedent : rule.antecedents)
            antecedentSet.add(antecedent);
        }
        antecedents = ArrayUtils.asPrimitiveIntArray(antecedentSet);
        for(int i = 0; i < antecedents.length; i++)
          antecedentMap.put(antecedents[i], i);
      }

      this.name = name;
      this.rules = rules;
      this.prior = prior;
      this.antecedents = antecedents;
      this.antecedentMap = antecedentMap;
      this.consequent = consequent;
      // Initialize components
      components = new ArrayList<>();
      for(int antecedent : antecedents)
        components.add(antecedent);
      components.add(consequent);
    }

    // TODO(Arun): Return the sorted list of explanations?
    public GroundedRule explain(boolean[] assignment) {
      double maxScore = Double.NEGATIVE_INFINITY;
      if(rules.size() == 0) return prior;
      GroundedRule bestRule = rules.get(0);
      for(GroundedRule rule : rules) {
        boolean match = true;
        for(int antecedent: rule.antecedents)
          match = match & assignment[antecedent];
        if(match) {
          double score = (assignment[consequent]) ? rule.logProbTrue : rule.logProbFalse;
          if(score > maxScore) {
            maxScore = score;
            bestRule = rule;
          }
        }
      }
      return bestRule;
    }
    public String getExplanation(boolean[] assignment) {
      StringBuilder sb = new StringBuilder();
      if(rules.size() == 0) sb.append(prior.name).append("\n");
      for(GroundedRule rule : rules) {
        boolean match = true;
        for(int antecedent: rule.antecedents)
          match = match & assignment[antecedent];
        if(match) {
          sb.append(rule.name).append("\n");
        }
      }
      return sb.toString().trim();
    }

    public double computeEntry(boolean[] assignment) {
      // Start adding.
      int sizeLimit = 0;
      int updates = 0;
      double logProbTrue = 0.;
      double logProbFalse = 0.;
      int[] updated = new int[]{0, 0};

      for(GroundedRule rule : rules) {
        // The list is sorted remember (gracefully falls back on to the prior)
        if( rule.antecedents.length < sizeLimit ) break;
        if( rule.antecedents.length == 0 ) assert rule.equals(prior);

        // Check if the rule matches
        boolean match = true;
        for(int antecedent: rule.antecedents)
          match = match & assignment[antecedent];
        if(!match) continue;
        sizeLimit = rule.antecedents.length;

        // Finally add yourself to logProbTrue.
        switch(Props.TEST_GRAPH_INFERENCE_RULE_MERGE_METHOD) {
          case HYBRID_OR: {
            // Take the noisy-or of the positively weighted rules and noisy-and of the negatively weighted rules and their mean.
            if(rule.logProbTrue > rule.logProbFalse) {
              // Prefer the truth
              double update = Math.min(prior.logProbFalse, rule.logProbFalse);
              updated[0] = 1;
              logProbFalse += update;
            } else {
              // Prefer the lie
              double update = Math.max(prior.logProbTrue, rule.logProbTrue);
              updated[1] = 1;
              logProbTrue += update;
            }
            break;
          }
          case GENTLE_OR: {
            // Take the noisy-or of the positively weighted rules and noisy-or of the negatively weighted rules and their mean (very forgiving).
            if(rule.logProbTrue > rule.logProbFalse) { // Positively weighted
              // Prefer the truth
              double update = Math.min(prior.logProbFalse, rule.logProbFalse);
              updated[0] = 1;
              logProbFalse += update;
            } else { // Negatively weighted
              // Prefer the lie
              double update = Math.min(prior.logProbFalse, rule.logProbFalse);
              updated[1] = 1;
              // HACK: Sneaking in the noisy-or of the negatively weighted rules into logProbTrue
              logProbTrue += update;
            }
            break;
          }
          case NOISY_OR: {
            double update = Math.min(prior.logProbFalse, rule.logProbFalse);
            logProbFalse += update;
            break;
          }
          case GEOMETRIC_MEAN: {
            // Use the mean
            double update = Math.max(prior.logProbTrue, rule.logProbTrue);
            logProbTrue += (update - logProbTrue)/(++updates);
            break;
          }
        }
      }
      // Finally compute the merged probability scores.
      switch(Props.TEST_GRAPH_INFERENCE_RULE_MERGE_METHOD) {
        case HYBRID_OR: {
          // Use the arithmethic mean of the trues and the fales
          logProbTrue = Math.log(
                  (updated[0] * (1. - Math.exp(logProbFalse)) +
                          updated[1] * Math.exp(logProbTrue))
                  / (updated[0] + updated[1]) );
          break;
        }
        case GENTLE_OR: {
          // Use the arithmethic mean of the trues and the fales
          logProbTrue = Math.log(
                  (updated[0] * (1. - Math.exp(logProbFalse)) +
                          updated[1] * (1. - Math.exp(logProbTrue)))
                          / (updated[0] + updated[1]) );
          break;
        }
        case NOISY_OR: {
          // Already did for logProbFalse
          logProbTrue = Math.log(1. - Math.exp(logProbFalse));
          break;
        }
        case GEOMETRIC_MEAN: {
          // Nothing to do!
          break;
        }
      }

      return logProbTrue;
    }

    public static void cleanPriors(List<GroundedRule> rules) {
      // Isolate the prior.
      assert rules.size() > 0;
      int consequent = rules.iterator().next().consequent;

      // Find or set the prior; choose the prior of the largest weight.
      GroundedRule prior = GroundedRule.empty(consequent);
      {
        double priorLogProbTrue = Double.NEGATIVE_INFINITY;
        for(GroundedRule rule : rules) {
          if( rule.antecedents.length != 0 ) continue;
          if(priorLogProbTrue < rule.logProbTrue) {
            priorLogProbTrue = rule.logProbTrue;
            prior = rule;
          }
        }
      }
      {
        // Remove the priors
        Iterator<GroundedRule> it = rules.iterator();
        while(it.hasNext()) {
          GroundedRule rule = it.next();
          if( rule.antecedents.length == 0 ) it.remove();
        }
        // Put the prior in the front
        rules.add(0, prior);
      }
    }

    @Override
    public String getName() { return name; }

    static long bitvectorToLong(boolean[] bitvector) {
      int res = 0;
      for (boolean bit : bitvector) {
        res = (res << 1) + (bit ? 1 : 0);
      }
      return res;
    }
    static long bitvectorToLong(boolean[] parent, int[] selectors) {
      int res = 0;
      for (int selector : selectors) {
        res = (res << 1) + (parent[selector] ? 1 : 0);
      }

      return res;
    }
    static void intToBitvector(long value, boolean[] bitvector) {
      assert value < (1 << bitvector.length);
      for(int i = bitvector.length-1; i >= 0; i--) {
        bitvector[i] = (value & 1) == 1;
        value >>= 1;
      }
    }

    String toBitString(boolean[] assignment) {
      char[] chars = new char[antecedents.length];
      for(int i = 0; i < antecedents.length; i++)
        chars[i] = assignment[antecedents[i]] ? '1' : '0';
      return new String(chars);
    }

    @Override
    public double logProb(boolean[] assignment) {
      String bitString = toBitString(assignment);
      if(!memoized.containsKey(bitString))
        memoized.put(bitString, computeEntry(assignment));
      double logProbTrue = memoized.get(bitString);

      if(assignment[consequent])
        return logProbTrue;
      else
        return Math.log(1. - Math.exp(logProbTrue));
    }
    @Override
    public Collection<Integer> components() {
      return components;
    }

    public String toString(){
      return name;
    }
  }

  protected static class EagerTableFactor extends TableFactor {

    double[] logProbTrue;
    double[] logProbFalse;

    void constructTable() {
      int entries = (1 << antecedents.length);
      int maxAssignment = 0;
      for(int antecedent : antecedents)
        maxAssignment = Math.max(maxAssignment, antecedent);

      boolean[] assignmentVector = new boolean[maxAssignment+1];
      boolean[] bv = new boolean[antecedents.length];

      assert prior.antecedents.length == 0;
      // Update prior
      for(int entry = 0; entry < entries; entry++) {
        intToBitvector(entry, bv);
        for(int i = 0; i < antecedents.length; i++)
          assignmentVector[antecedents[i]] = bv[i];
        logProbTrue[entry] = computeEntry(assignmentVector);
        logProbFalse[entry] = Math.log(1. - Math.exp(logProbTrue[entry]));

        for(int i = 0; i < antecedents.length; i++)
          assignmentVector[antecedents[i]] = false;
      }
    }

    public EagerTableFactor(List<GroundedRule> rules) {
      super(rules);
      assert(antecedents.length < 16);
      // Construct a table eagerly.
      // Create the mega-table;
      int entries = (1 << antecedents.length);
      logProbTrue = new double[entries];
      logProbFalse = new double[entries];
      constructTable();
    }
    public EagerTableFactor(TableFactor factor) {
      super(factor);
      assert(antecedents.length < 12);
      // Construct a table eagerly.
      // Create the mega-table;
      int entries = (1 << antecedents.length);
      logProbTrue = new double[entries];
      logProbFalse = new double[entries];
      constructTable();
    }

    @Override
    public double logProb(boolean[] assignment) {
      long selector = bitvectorToLong(assignment, antecedents);
      if(assignment[consequent])
        return logProbTrue[(int)selector];
      else
        return logProbFalse[(int)selector];
    }
  }

  // Factors and variables
  private Collection<BayesNet.Factor> factors = new ArrayList<>();
  private Index<MLNText.Literal> variableIndexer = new HashIndex<>();
  private Map<Integer, Boolean> fixedValues = new HashMap<>();
  private Map<Integer,Double> priors = new HashMap<>();
  private List<MLNText.Rule> priorRules = new ArrayList<>();

  // Utilities for grounding
  private MLNText program = new MLNText();
  private Map<String, Collection<String>> domains = new HashMap<>();
  private Map<MLNText.Predicate, List<MLNText.Literal>> closedWorldEvidence = new HashMap<>();
  private Map<Integer, List<GroundedRule>> factorsForLiteral = new HashMap<>();

  protected static class GroundedRule {
    final String name;
    final double logProbTrue;
    final double logProbFalse;
    final int consequent;
    final int[] antecedents;

    GroundedRule(String name, double logProbTrue, double logProbFalse, int consequent, int... antecedents) {
      this.name = name;
      this.logProbTrue = logProbTrue;
      this.logProbFalse = logProbFalse;
      this.consequent = consequent;
      this.antecedents = antecedents;
    }

    public static GroundedRule empty(int consequent) {
      return new GroundedRule("default prior", Math.log(Props.TEST_GRAPH_INFERENCE_PRIOR), Math.log(1.0 - Props.TEST_GRAPH_INFERENCE_PRIOR), consequent);
    }

    public MLNText.Rule toRule(MLNText.Literal[] literals) {
      MLNText.Rule rule = new MLNText.Rule();
      for(int antecedent : antecedents)
        rule.literals.add(literals[antecedent].asFalse());
      rule.literals.add(literals[consequent].asTrue());
      rule.weight = logProbTrue - logProbFalse;
      return rule;
    }
  }

  // Parameters
  private boolean doHillclimb = false;

  protected boolean isConstant(String arg) {
    return Character.isUpperCase(arg.charAt(0));
  }

  // Overrides for the logProbTrue, logProb false;
  protected void registerRuleInstance(double logProbTrue, double logProbFalse, final MLNText.Literal[] groundedLiterals) {
    // Note: Not error checking because post modifying rule weights, things can no longer be probabilities.

    MLNText.Rule rule = new MLNText.Rule(logProbTrue - logProbFalse, Arrays.asList(groundedLiterals) );
    logProbTrue = clipLogProb(rule.toString(), logProbTrue);
    logProbFalse = clipLogProb(rule.toString(), logProbFalse);


    // Activate all the literals
    int consequent = -1;
    int[] antecedents = new int[groundedLiterals.length-1];
    {
      int i = 0;
      for( MLNText.Literal literal : groundedLiterals) {
        // HACK
        assert(isConstant(literal.arg1));
        assert(isConstant(literal.arg2));
        if(literal.truth) {
          consequent =  variableIndexer.addToIndex(literal.asTrue());
        } else {
          antecedents[i++] = variableIndexer.addToIndex(literal.asTrue());
        }
      }
    }
    assert consequent != -1;

    if(!factorsForLiteral.containsKey(consequent))
      factorsForLiteral.put(consequent, new ArrayList<GroundedRule>());
    factorsForLiteral.get(consequent).add(new GroundedRule(rule.toString(), logProbTrue, logProbFalse, consequent, antecedents));
  }

  /**
   * Recursively ground the provided rule template
   * Note:  A capital argument is a ground variable.
   * @param rule - Rule template
   * @param groundedLiterals - partially grounded literals
   * @param binding - variable binding
   * @param index - grounding progress
   */
  private void registerRuleTemplateHelper(final MLNText.Rule rule, double logProbTrue, double logProbFalse, final MLNText.Literal[] groundedLiterals, final Map<String, String> binding, final int index) {
    if (index >= groundedLiterals.length) {
      registerRuleInstance(logProbTrue, logProbFalse, groundedLiterals);
    } else {
      // Try to bind this relation.
      boolean[] backtrackBinding = {false, false};
      MLNText.Literal literal = rule.literals.get(index);
      MLNText.Predicate pred = program.getPredicateByName(literal.name).orCrash();

      if(pred.closed) {
        // Go through all the evidence of this closed type and attempt to ground it.
        for(MLNText.Literal groundedLiteral : closedWorldEvidence.get(pred) ) {
          // Attempt to bind 1
          if(isConstant(literal.arg1)) {
            if(!groundedLiteral.arg1.equals(literal.arg1)) continue;
          } else if(binding.containsKey(literal.arg1)) {
            if(!groundedLiteral.arg1.equals(binding.get(literal.arg1))) continue;
          } else {
            backtrackBinding[0] = true;
          }

          if(isConstant(literal.arg2)) {
            if(!groundedLiteral.arg2.equals(literal.arg2)) continue;
          } else if(binding.containsKey(literal.arg2)) {
            if(!groundedLiteral.arg2.equals(binding.get(literal.arg2))) continue;
          } else {
            backtrackBinding[1] = true;
          }


          if(backtrackBinding[0]) {
            if(binding.containsValue(groundedLiteral.arg1)) continue;
            binding.put(literal.arg1, groundedLiteral.arg1);
          }
          if(backtrackBinding[1]) {
            if(binding.containsValue(groundedLiteral.arg2)) continue;
            binding.put(literal.arg2, groundedLiteral.arg2);
          }

          // Store this as the current literal and proceed.
          groundedLiterals[index] = literal.withArgs(groundedLiteral.arg1, groundedLiteral.arg2);
          registerRuleTemplateHelper(rule, logProbTrue, logProbFalse, groundedLiterals, binding, index+1);
          if(backtrackBinding[0])
            binding.remove(literal.arg1);
          if(backtrackBinding[1])
            binding.remove(literal.arg2);
        }
      } else {
        // Ground literals of index
        String type1 = pred.type1;
        String type2 = pred.type2;

        // Error checking
        if(domains.get(type1) == null) {
          warn("No entities registered of type : " + type1 + "; see registerDomain()");
          return;
        }
        if(domains.get(type2) == null) {
          warn("No entities registered of type : " + type2 + "; see registerDomain()");
          return;
        }
        // Is this variable bound?
        Collection<String> candidates1;
        if(isConstant(literal.arg1)) {// Capital letter => ground constant
          candidates1 = Collections.singleton(literal.arg1);
        } else if (binding.containsKey(literal.arg1)) {
          candidates1 = Collections.singleton(binding.get(literal.arg1));
        } else {
          candidates1 = domains.get(type1);
          backtrackBinding[0] = true;
        }
        Collection<String> candidates2;
        if(isConstant(literal.arg2)) {// Capital letter => ground constant
          candidates2 = Collections.singleton(literal.arg2);
        } else if (binding.containsKey(literal.arg2)) {
          candidates2 = Collections.singleton(binding.get(literal.arg2));
        } else {
          candidates2 = domains.get(type2);
          backtrackBinding[1] = true;
        }

        for(String arg1 : candidates1) {
          for(String arg2 : candidates2) {
            // Make sure this is a valid pairing
            if(validPairings.containsKey(arg1) && !validPairings.get(arg1).contains(arg2)) continue;

            // Variable names are unique! so if you've bound this before, don't.
            if(backtrackBinding[0]) {
              if(binding.containsValue(arg1)) continue;
              binding.put(literal.arg1, arg1);
            }
            if(backtrackBinding[1]) {
              if(binding.containsValue(arg2)) continue;
              binding.put(literal.arg2, arg2);
            }

            // Store this as the current literal and proceed.
            groundedLiterals[index] = literal.withArgs(arg1, arg2);
            // Recursive call -- eek intractable! Grounding is hard...
            registerRuleTemplateHelper(rule, logProbTrue, logProbFalse, groundedLiterals, binding, index+1);
            if(backtrackBinding[0])
              binding.remove(literal.arg1);
            if(backtrackBinding[1])
              binding.remove(literal.arg2);
          }
        }
      }
    }
  }

  /**
   * Recursively ground the provided rule template.
   * @param rule
   * @return
   */
  protected BayesNetBuilder registerRuleTemplate(MLNText.Rule rule, double logProbTrue, double logProbFalse) {
    // Copy rule
    MLNText.Literal[] literals = new MLNText.Literal[rule.literals.size()];
    Map<String, String> binding = new HashMap<>();
    registerRuleTemplateHelper(rule, logProbTrue, logProbFalse, literals, binding, 0);
    return this;
  }

  public BayesNetBuilder registerDomain(String name) {
    if(!domains.containsKey(name))
      domains.put(name, new HashSet<String>());
    return this;
  }
  public BayesNetBuilder registerDomain(String name, Collection<String> values) {
    domains.put(name, values);
    return this;
  }
  public BayesNetBuilder registerConstant(String typeName, String value) {
    if(!domains.containsKey(typeName)) registerDomain(typeName);
    domains.get(typeName).add(value);
    return this;
  }

  /**
   * Add a closed world literal
   * @param literal
   * @return
   */
  public BayesNetBuilder registerEvidence(MLNText.Literal literal) {
    // Create the predicate if it hasn't already been created
    MLNText.Predicate pred = program.getPredicateByName(literal.name).orCrash();
    assert pred.closed;
    // Set as things that are fixed true / false
    closedWorldEvidence.get(pred).add(literal);
    {
      int idx = variableIndexer.addToIndex(literal.asTrue());
      fixedValues.put(idx, literal.truth);
    }

    return this;
  }

  public BayesNetBuilder registerPredicate(MLNText.Predicate predicate) {
    if(!program.predicates.contains(predicate)) {
      program.predicates.add(predicate);
      registerDomain(predicate.type1);
      registerDomain(predicate.type2);
      if(predicate.closed)
        closedWorldEvidence.put(predicate, new ArrayList<MLNText.Literal>());
    }
    return this;
  }

  public BayesNetBuilder paramDoHillClimb(boolean doHillclimb) {
    this.doHillclimb = doHillclimb;
    return this;
  }

  public BayesNetBuilder addPredicates(Collection<MLNText.Predicate> predicates) {
    for(MLNText.Predicate predicate : predicates)
      registerPredicate(predicate);
    return this;
  }

  /**
   * Must have added predicates before this.
   * TODO(Arun): Make less specific to KBP
   * @param constants
   * @return
   */
  public BayesNetBuilder addConstants(Map<String, KBPEntity> constants) {
    for(Map.Entry<String,KBPEntity> constant : constants.entrySet())
      registerConstant(constant.getValue().type.toString(), constant.getKey());
    return this;
  }

  /**
   * Must have added predicates before this.
   * @param evidence
   * @return
   */
  public BayesNetBuilder addEvidence(Collection<MLNText.Literal> evidence) {
    for(MLNText.Literal literal : evidence)
      registerEvidence(literal);
    return this;
  }
  public BayesNetBuilder addPrior(MLNText.Rule rule) {
    this.priorRules.add(rule);
    return this;
  }
  public BayesNetBuilder addPriors(Collection<MLNText.Rule> rules) {
    this.priorRules.addAll(rules);
    return this;
  }
  public BayesNetBuilder addRule(MLNText.Rule rule) {
    for(MLNText.Literal literal : rule.literals) {
      MLNText.Predicate pred = program.getPredicateByName(literal.name).orCrash();

      // If the rule contains any constants, add them too
      if(isConstant(literal.arg1)) registerConstant(pred.type1, literal.arg1);
      if(isConstant(literal.arg2)) registerConstant(pred.type2, literal.arg2);
    }
    program.rules.add(rule);
    return this;
  }
  public BayesNetBuilder addRules(Collection<MLNText.Rule> rules) {
    for(MLNText.Rule rule : rules)
      addRule(rule);
    return this;
  }
  public BayesNetBuilder addMLN(MLNText mln) {
    addPredicates(mln.predicates);
    addRules(mln.rules);
    return this;
  }
  public BayesNetBuilder setValidPairings(Map<String,Set<String>> validPairings) {
    this.validPairings = validPairings;
    return this;
  }

  /**
   * Ground all rules and create a BayesNet
   * @return
   */
  public BayesNet<MLNText.Literal> build(){
    // Put in all the prior rules
    for(MLNText.Rule rule : priorRules ) {
      double prob = SloppyMath.sigmoid(rule.weight);
      double logProbTrue = Math.log(prob);
      double logProbFalse = Math.log(1 - prob);

      assert rule.literals.size() == 1;
      MLNText.Literal literal = rule.literals.get(0);
      registerRuleInstance(logProbTrue, logProbFalse, new MLNText.Literal[]{literal});
    }
    // Ground all with length > 1 rules
    for(MLNText.Rule rule: program.rules) {
      // Add the priors in later
      if(rule.literals.size() == 1) continue;
      double prob = SloppyMath.sigmoid(rule.weight);
      double logProbTrue = Math.log(prob);
      double logProbFalse = Math.log(1 - prob);
      registerRuleTemplate(rule, logProbTrue, logProbFalse);
    }
    // Now ground all the priors only on the literals you've considered so far.
    for(MLNText.Rule rule: program.rules) {
      if(rule.literals.size() != 1) continue;
      double prob = SloppyMath.sigmoid(rule.weight);
      double logProbTrue = Math.log(prob);
      double logProbFalse = Math.log(1 - prob);

      MLNText.Literal priorLiteral = rule.literals.get(0);

      // For every literal we've grounded so far, add this prior
      for(MLNText.Literal literal : variableIndexer.objectsList()) {
        if(literal.name.equals(priorLiteral.name)) {
          registerRuleInstance(logProbTrue, logProbFalse, new MLNText.Literal[]{priorLiteral.withArgs(literal.arg1, literal.arg2)});
        }
      }
    }

    // Finally collapse into factors.
    for( Map.Entry<Integer,List<GroundedRule>> entry : factorsForLiteral.entrySet()) {
      int variableId = entry.getKey();
      List<GroundedRule> implications  = entry.getValue();
      // If a variable only has false variables, skip it!
      if(CollectionUtils.all(implications, in -> in.logProbTrue < Math.log(0.3))) {
        // this variable will always be false. Set it so.
        fixedValues.put(variableId, false);
      } else  if(CollectionUtils.all(implications, in -> in.logProbTrue > Math.log(0.9))) {
          // this variable will always be false. Set it so.
          fixedValues.put(variableId, true);
      } else {
        factors.addAll(makeFactors(implications));
      }
    }

    return new BayesNet<>(
            variableIndexer,
            variableIndexer.objectsList().toArray(new MLNText.Literal[variableIndexer.size()]),
            factors.toArray(new BayesNet.Factor[factors.size()]),
            priors,
            fixedValues,
            this.doHillclimb);
  }

  /**
   * HACK: Do whatever it takes to reduce the number of antecedents to a reasonable amount.
   * @return
   */
  private Collection<? extends BayesNet.Factor> makeFactors(List<GroundedRule> rules) {
    // Make unique
    rules = new ArrayList<>(new HashSet<>(rules));

    TableFactor.cleanPriors(rules);
    if(rules.size() == 1) return Collections.singleton(new EagerTableFactor(rules));
    GroundedRule prior = rules.remove(0);
    priors.put(prior.consequent, prior.logProbTrue);

    // Start scrolling through rules until while you can batch them up into sets of < 12.
    List<EagerTableFactor> factors = new ArrayList<>();
    List<GroundedRule> buffer = new ArrayList<>();
    Set<Integer> antecedents = new HashSet<>();

    // Shuffle the rules
    Collections.shuffle(rules, new Random(42));

    for(GroundedRule rule : rules) {
      for(int antecedent : rule.antecedents)
        antecedents.add(antecedent);
      // Peek - what happens if I add this
      if(antecedents.size() > 12) {
        // clear the buffer
        buffer.add(0, prior);
        factors.add(new EagerTableFactor(buffer));
        buffer.clear();
        antecedents.clear();
        // And re-add
        for(int antecedent : rule.antecedents)
          antecedents.add(antecedent);
      } else {
        buffer.add(rule);
      }
    }
    if(!buffer.isEmpty()) {
      buffer.add(0, prior);
      factors.add(new EagerTableFactor(buffer));
    }

    return factors;
  }

  /**
   * Takes a list of MLN rules and attempts to choose the ones that
   * have the maximum likelihood under the data and form a valid
   * Bayesian logic network (i.e. make formation of loops impossible)
   * @param rules - Candidate list of rules
   * @return - A subset of the rules that is acyclic
   */
  public static MLNText makeAcyclic(MLNText rules) {
    startTrack("Making rules acyclic");
    logf("Starting with %d rules", rules.rules.size());
    // Sort the rules by their weights.
    Collections.sort(rules.rules, (o1, o2) -> {
      // TODO(arun) : Modify to include prior weights or otherwise convert this to be sensible
      if (o1.weight < o2.weight) return 1;
      else if (o1.weight == o2.weight) return 0;
      else return -1;
    });

    // Now greedily introduce rules
    final Map<String, Set<String>> ancestors = new HashMap<>();
    for(MLNText.Predicate pred : rules.predicates) {
      ancestors.put(pred.name, new HashSet<String>());
      ancestors.get(pred.name).add(pred.name);
    }
    MLNText validRules = new MLNText();
    for(MLNText.Rule rule : rules.rules) {
      final Set<String> consequents = new HashSet<>();
      final Set<String> antecedents = new HashSet<>();
      for(MLNText.Literal in : rule.literals) {
        if(in.truth) consequents.add(in.name);
        else antecedents.add(in.name);
      }

      // Does this produces a loop?
      if( CollectionUtils.any(antecedents, in -> CollectionUtils.overlap(ancestors.get(in), consequents).isDefined()) ) {
        debug("Excluding rule " + rule);
        continue;
      }

      // No? Good, add it.
      // Add all the predicates
      for(MLNText.Literal literal : rule.literals) {
        validRules.predicates.add(rules.getPredicateByName(literal.name).get());
      }
      validRules.add(rule);

      // Update the ancestors list
      for(String consequent : consequents) {
        for(String antecedent : antecedents)
          ancestors.get(consequent).addAll(ancestors.get(antecedent));
      }
    }
    logf("Ending with %d rules", validRules.rules.size());
    endTrack("Making rules acyclic");

    return validRules;
  }

}
