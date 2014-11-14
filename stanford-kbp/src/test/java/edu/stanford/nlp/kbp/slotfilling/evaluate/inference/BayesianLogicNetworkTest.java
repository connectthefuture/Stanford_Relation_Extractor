package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import org.junit.Test;

import static edu.stanford.nlp.util.logging.Redwood.Util.log;

/**
 * Tests for Bayesian logic networks
 */
public class BayesianLogicNetworkTest {

  @Test
  public void testMakeAcyclic() {
    MLNText rules = new MLNTextBuilder()
            .addPredicates()
            .openPredicate("likes", "PESRON", "PLACE")
            .openPredicate("lives in", "PERSON", "PLACE")
            .openPredicate("works at", "PESRON", "ORGANIZATION")
            .openPredicate("headquartered at", "ORGANIZATION", "PLACE")
            .endPredicates()
            .addRules()
            .newRule().orNot("likes", "x0", "x1").or("lives in", "x0", "x1").endRule(0.9)
            .newRule().orNot("lives in", "x0", "x1").or("likes", "x0", "x1").endRule(0.9)
            .newRule().orNot("works at", "x0", "x1").orNot("headquartered at", "x1", "x2").or("likes", "x0", "x2").endRule(0.9)
            .endRules()
            .end();

    BayesianLogicNetwork bln = BayesianLogicNetwork.buildAcyclic(rules);
    log(bln.rules);
  }

}
