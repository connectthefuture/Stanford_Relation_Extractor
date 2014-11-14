package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.util.StringUtils;

import java.util.*;

/**
* A textual representation of an MLN
*/
public class MLNText {
  public static class Predicate {
    final boolean closed;
    final String name;
    final String type1;
    final String type2;

    // TODO(arun): Extend to more than 2 args
    public Predicate(boolean closed, String name, String type1, String type2) {
      this.closed = closed;
      this.name = name;
      this.type1 = type1;
      this.type2 = type2;
    }
    public Predicate(String name, String type1, String type2) {
      this(false, name, type1, type2);
    }

    public String toString() {
      return String.format( "%s%s(%s,%s)", closed ? "*" : "", name, type1, type2);
    }

    @Override
    public int hashCode() {
      return toString().hashCode();
    }

    @Override
    public boolean equals(Object other_) {
      if(other_ instanceof Predicate) {
        return other_.toString().equals(toString());
      }
      return false;
    }

    public Predicate asOpen() {
      return new Predicate(false, name, type1, type2);
    }
    public Predicate asClosed() {
      return new Predicate(true, name, type1, type2);
    }
  }
  public static class Literal {
    final String name;
    final String arg1;
    final String arg2;
    final boolean truth;

    public Literal(boolean truth, String name, String arg1, String arg2) {
      this.truth = truth;
      this.name = name;
      this.arg1 = arg1;
      this.arg2 = arg2;
    }

    public Literal withArgs(String arg1, String arg2) {
      return new Literal(truth, name, arg1, arg2);
    }

    public String toString() {
      if(truth)
        return String.format( "%s(%s,%s)", name, arg1, arg2);
      else
        return String.format( "!%s(%s,%s)", name, arg1, arg2);
    }

    @Override
    public int hashCode() {
      return toString().hashCode();
    }

    @Override
    public boolean equals(Object other_) {
      if(other_ instanceof Literal) {
        return other_.toString().equals(toString());
      }
      return false;
    }
    public Literal asTrue() {
      return new Literal(true, name, arg1, arg2);
    }

    public Literal asFalse() {
      return new Literal(false, name, arg1, arg2);
    }
  }
  public static class Rule {
    List<Literal> literals;
    double weight;

    public Rule() {
      weight = Double.POSITIVE_INFINITY;
      literals = new ArrayList<>();
    }
    public Rule(double weight, List<Literal> literals) {
      this.weight = weight;
      this.literals = new ArrayList<>(literals);
    }
    public Rule(double weight, Literal... literals) {
      this.weight = weight;
      this.literals = new ArrayList<>(Arrays.asList(literals));
    }

    public String toString() {
      if(Double.isInfinite(weight))
        return StringUtils.join(literals, " v ") + ".";
      else
        return weight + " " + StringUtils.join(literals, " v ");
    }

    @Override
    public int hashCode() {
      return toString().hashCode();
    }

    @Override
    public boolean equals(Object other_) {
      if(other_ instanceof Rule) {
        Rule other = (Rule) other_;
        return (other.weight == weight) && other.literals.equals(literals);
      }
      return false;
    }

    public List<Literal> antecedents() {
      List<Literal> antecedents = new ArrayList<>();
      for(Literal literal : literals)
        if(!literal.truth) antecedents.add(literal);
      return antecedents;
    }
    public Literal consequent() {
      for(Literal literal : literals)
        if(literal.truth) return literal;
      throw new IndexOutOfBoundsException();
    }
  }

  Set<Predicate> predicates = new HashSet<>();
  List<Rule> rules = new ArrayList<>();

  public MLNText() {}
  public MLNText(MLNText other) {
    this.predicates = new HashSet<>(other.predicates);
    this.rules = new ArrayList<>(other.rules);
  }

  public void add(Rule rule) {
    this.rules.add(rule);
  }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    for(Predicate predicate : predicates)
      sb.append(predicate).append("\n");
    sb.append("\n");
    for(Rule rule : rules)
      sb.append(rule).append("\n");
    return sb.toString();
  }

  public boolean equals(Object other_) {
    if( other_ instanceof MLNText) {
      MLNText other = (MLNText) other_;
      return other.predicates.equals(predicates) && other.rules.equals(rules);
    }
    return false;
  }

  public void mergeIn(MLNText other) {
    this.predicates.addAll(other.predicates);
    this.rules.addAll(other.rules);
  }

  public static MLNText union(MLNText first, MLNText second) {
    MLNText text = new MLNText();
    text.predicates.addAll(first.predicates);
    text.predicates.addAll(second.predicates);

    text.rules.addAll(first.rules);
    text.rules.addAll(second.rules);

    return text;
  }

  public Maybe<Predicate> getPredicateByName(String relationName) {
    for(Predicate pred : predicates)
      if( pred.name.equals(relationName) )
        return Maybe.Just(pred);
    return Maybe.Nothing();
  }

  public MLNText withoutPredicates() {
    MLNText text = new MLNText();
    text.rules.addAll(this.rules);
    return text;
  }
}
