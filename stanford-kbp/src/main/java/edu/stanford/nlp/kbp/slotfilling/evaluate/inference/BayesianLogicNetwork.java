package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.kbp.common.CollectionUtils;
import java.util.function.Function;

import java.util.*;

/**
 * Represents a Bayesian logic network
 */
public class BayesianLogicNetwork {
  final public MLNText rules;

  public BayesianLogicNetwork(MLNText rules) {
    this.rules = rules;
  }

  /**
   * Build a BLN from a possibly acyclic rules file.
   * @param rules - A possible acyclic set of rules
   * @return - A Bayesian Logic network.
   */
  public static BayesianLogicNetwork buildAcyclic(MLNText rules) {
    return new BayesianLogicNetwork(makeAcyclic(rules));
  }

  /**
   * Takes a list of MLN rules and attempts to choose the ones that
   * have the maximum likelihood under the data and form a valid
   * Bayesian logic network (i.e. make formation of loops impossible)
   * @param rules - Candidate list of rules
   * @return - A subset of the rules that is acyclic
   */
  public static MLNText makeAcyclic(MLNText rules) {
    // Sort the rules by their weights.
    Collections.sort(rules.rules, (o1, o2) -> {
      // TODO(arun) : Modify to include prior weights or otherwise convert this to be sensible
      if (o1.weight > o2.weight) return 1;
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
      if( CollectionUtils.any( antecedents, in -> CollectionUtils.overlap(ancestors.get(in), consequents).isDefined()) ) continue;

      // No? Good, add it.
      validRules.add(rule);

      // Update the ancestors list
      for(String consequent : consequents) {
        for(String antecedent : antecedents)
          ancestors.get(consequent).addAll(ancestors.get(antecedent));
      }
    }

    return validRules;
  }

}
