package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Creates a MLN Text object
 */
public class MLNTextBuilder {
  MLNText mlnText = new MLNText();

  public PredicateBuilder addPredicates() { return this.new PredicateBuilder(); }

  public RuleBuilder addRules() { return new RuleBuilder(); }

  public MLNText end() { return mlnText; }

  
  public void addPredicate(MLNText.Predicate pred) {
    mlnText.predicates.add(pred);
  }

  //
  //  --------------------
  //  Register Predicates
  //  --------------------
  //
  /**
   * A class to provide the interface for the predicate declaration phase of creating the network.
   */
  public class PredicateBuilder {
    private PredicateBuilder() { }

    /**
     * Register a predicate.
     * @param closed True if this is a closed-world predicate.
     * @param name The name of the predicate.
     * @param argTypes The types of the arguments of the predicate
     * @return this
     */
    private PredicateBuilder predicate(boolean closed, String name, String... argTypes) {
      mlnText.predicates.add( new MLNText.Predicate(name, argTypes[0], argTypes[1]) );
      return this;
    }

    /**
     * Register a closed-world predicate.
     * That is, the only true instantiations of the predicate are those defined in the evidence.
     * @param name The name of the predicate.
     * @param argTypes The types of the arguments of the predicate
     * @return this
     */
    public PredicateBuilder closedPredicate(String name, String... argTypes) { return predicate(true, name, argTypes); }
    /**
     * Register a open-world predicate.
     * @param name The name of the predicate.
     * @param argTypes The types of the arguments of the predicate
     * @return this
     */
    public PredicateBuilder openPredicate(String name, String... argTypes) { return predicate(false, name, argTypes); }

    public MLNTextBuilder endPredicates() {
      return MLNTextBuilder.this;
    }
  }

  //
  //  --------------------
  //  Register Rules
  //  --------------------
  //

  /**
   * The interface for creating a single rule. The end responsibility of this class is to create a Clause
   * and add it to the Markov Logic Net
   */
  public class SingleRuleBuilder {
    /** The clause to build */
    private final ArrayList<MLNText.Literal> literals = new ArrayList<>();
    /** The user-specified name for the rule */
    private final String name;

    private SingleRuleBuilder() { this.name = null; }
    private SingleRuleBuilder(String name) { this.name = name; }

    /**
     * The internal implementation of adding an "or" term (single predicate) into the clause.
     * @param negate If true, this predicate is negated.
     * @param predicate The predicate name to add.
     * @param variables The predicate variables (or constants) to include.
     * @return this
     */
    private SingleRuleBuilder or(boolean negate, String predicate, String... variables) {
      literals.add( new MLNText.Literal(!negate, predicate, variables[0], variables[1]));
      return this;
    }

    /**
     * Enforce two terms to be '='
     * @param variable1 The predicate variables (or constants) to include.
     * @param variable2 The predicate variables (or constants) to include.
     * @return this
     */
    public SingleRuleBuilder equals(String variable1, String variable2 ) {
      return or(true, "=", variable1, variable2);
    }

    public SingleRuleBuilder notEquals(String variable1, String variable2 ) {
      return or(false, "=", variable1, variable2);
    }

    /**
     * Add another predicate to this or clause, not negated. For example, add Foo(x) to yield (Bar(x) v Foo(x)).
     * @param predicate The predicate name to add.
     * @param variables The predicate variables (or constants) to include.
     * @return this
     */
    public SingleRuleBuilder or(String predicate, String... variables) { return or(false, predicate, variables); }
    /**
     * Add a negated predicate to this or clause, not negated. For example, add !Foo(x) to yield (Bar(x) v !Foo(x)).
     * @param predicate The predicate name to add.
     * @param variables The predicate variables (or constants) to include.
     * @return this
     */
    public SingleRuleBuilder orNot(String predicate, String... variables) { return or(true, predicate, variables); }

    /**
     * Finish constructing this rule, and add it to the list of rules in the network.
     * @param weight The weight to add this rule with.
     * @return this
     */
    public RuleBuilder endRule(double weight) {
      if( literals.size() > 0 ) {
        mlnText.rules.add(new MLNText.Rule(weight, literals));
      }
      return new RuleBuilder();
    }

    /** @see edu.stanford.nlp.kbp.slotfilling.evaluate.inference.MLNTextBuilder.SingleRuleBuilder#endRule()   */
    public RuleBuilder endRule() {
      return endRule(Double.POSITIVE_INFINITY);
    }
  }

  /**
   * A class to provide the interface for the rule declaration phase of creating the network.
   */
  public class RuleBuilder {
    private RuleBuilder() { }

    /**
     * Start constructing a new Rule to add to the Markov Logic Network.
     * You can close this rule by calling endRule()
     * @return this
     */
    public SingleRuleBuilder newRule() {
      return new SingleRuleBuilder();
    }

    /**
     *
     * Start constructing a new named Rule to add to the Markov Logic Network.
     * You can close this rule by calling endRule()
     * @param name The user-specified name for the rule. This should be unique
     * @return this
     */
    public SingleRuleBuilder newRule(String name) {
      return new SingleRuleBuilder(name);
    }

    public MLNTextBuilder endRules() {
      return MLNTextBuilder.this;
    }
  }

  /** Return the string. */
  public String toString() { return mlnText.toString(); }

}
