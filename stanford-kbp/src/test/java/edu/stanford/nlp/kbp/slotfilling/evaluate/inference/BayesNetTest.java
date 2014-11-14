package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.Execution;
import junit.framework.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.log;
import static junit.framework.Assert.*;

/**
 * A test for {@link BayesNet} and {@link BayesNetBuilder}.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("unchecked")
@Ignore
public class BayesNetTest {

  private static MLNText.Literal triple(boolean truth, String ent, String rel, String slot) {
    KBTriple triple = KBPNew
        .entName(ent.substring(0, ent.indexOf(":"))).entType(NERTag.fromShortName(ent.substring(ent.indexOf(":") + 1)).orCrash())
        .slotValue(slot.substring(0, slot.indexOf(":"))).slotType(NERTag.fromShortName(slot.substring(slot.indexOf(":") + 1)).orCrash())
        .rel(rel).KBTriple();
    return new MLNText.Literal(truth, rel, triple.entityName, triple.slotValue);
  }
  private static MLNText.Literal triple(String ent, String rel, String slot) {
    return triple(true, ent, rel, slot);
  }

  private static KBPEntity person(String name) {
    return KBPNew.entName(name).entType(NERTag.PERSON).KBPEntity();
  }

  private static KBPEntity country(String name) {
    return KBPNew.entName(name).entType(NERTag.COUNTRY).KBPEntity();
  }


  private static MLNText.Rule singleton(String ent, String rel, String slot) {
    return singleton(1.0, ent, rel, slot);
  }
  private static MLNText.Rule singleton(double prob, String ent, String rel, String slot) {
    double weight = Math.log(prob) - Math.log(1. - prob);
    return new MLNText.Rule(weight, Arrays.asList(triple(true, ent, rel, slot)));
  }

  private static MLNText.Rule binary(String ent1, String rel1, String slot1,
                                   String ent2, String rel2, String slot2) {
    return binary(1.0, ent1, rel1, slot1, ent2, rel2, slot2);
  }
  private static MLNText.Rule binary(double prob, String ent1, String rel1, String slot1,
                                     String ent2, String rel2, String slot2) {
    double weight = Math.log(prob) - Math.log(1. - prob);
    return new MLNText.Rule(weight, Arrays.asList( triple(false, ent1, rel1, slot1), triple(true, ent2, rel2, slot2) ));
  }

  private static void sanityCheck(BayesNet<KBTriple> net) {
    assertNotNull(net);
    boolean containsUnaries = false;
    for (BayesNet.Factor factor : net) {
      assertFalse(factor.components().isEmpty());
      if (factor.components().size() == 1) { containsUnaries = true; }
    }
    if (!containsUnaries) {
//      assertEquals(((double) net.size()) * Math.log(0.5), net.logProb(new HashSet<KBTriple>()), 1e-3);
    }
  }

  //
  // CONSTRUCTION
  //

  @Test
  public void testCanMakeEmptyBayesNet() {
    BayesNet net = new BayesNetBuilder().build();
    sanityCheck(net);
    assertEquals(0, net.size());
    assertEquals(0, net.variableCount());
  }

  @Test
  public void testBayesNetSingleUnaryPredicate() {
    BayesNet net = new BayesNetBuilder()
            .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
            .addPrior(singleton("Julie:PER", "likes", "Canada:CRY"))
            .build();
    sanityCheck(net);
    assertEquals(1, net.size());
    assertEquals(1, net.variableCount());
  }

  @Test
  public void testBayesNetSingleBinaryPredicate() {
    BayesNet net = new BayesNetBuilder()
            .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
            .registerPredicate(new MLNText.Predicate("origin", "PERSON", "COUNTRY"))
            .addRule(binary("Julie:PER", "origin", "Canada:CRY", "Julie:PER", "likes", "Canada:CRY"))
            .build();
    sanityCheck(net);
    assertEquals(1, net.size());
    assertEquals(2, net.variableCount());
  }

  @Test
  public void testBayesNetMultipleBinaryPredicate() {
    BayesNet net = new BayesNetBuilder()
        .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
        .registerPredicate(new MLNText.Predicate("origin", "PERSON", "COUNTRY"))
        .registerPredicate(new MLNText.Predicate("welcomes home", "COUNTRY", "PERSON"))
        .addRule(binary("Julie:PER", "origin", "Canada:CRY", "Julie:PER", "likes", "Canada:CRY"))
        .addRule(binary("Julie:PER", "likes", "Canada:CRY", "Canada:CRY", "welcomes home", "Julie:PER"))
        .build();
    sanityCheck(net);
    assertEquals(2, net.size());
    assertEquals(3, net.variableCount());
  }

  @Test
  public void testBayesNetMultipleUnaryPredicates() {
    BayesNet net = new BayesNetBuilder()
        .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
        .registerPredicate(new MLNText.Predicate("origin", "PERSON", "COUNTRY"))
        .registerPredicate(new MLNText.Predicate("is", "PERSON", "TITLE"))
        .addPrior(singleton("Julie:PER", "likes", "Canada:ORG"))
        .addPrior(singleton("Julie:PER", "origin", "Finnish:NAT"))
        .addPrior(singleton("Arun:PER", "is", "Student:TIT"))
        .build();
    sanityCheck(net);
    assertEquals(3, net.size());
    assertEquals(3, net.variableCount());
  }

  //
  // INFERENCE
  //
  @Test
  public void testBayesNetGibbsUnaryFactorsTrivial() {
    BayesNet net = new BayesNetBuilder()
        .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
        .registerPredicate(new MLNText.Predicate("origin", "PERSON", "COUNTRY"))
        .registerPredicate(new MLNText.Predicate("is", "PERSON", "TITLE"))
        .addPrior(singleton(1.0, "Julie:PER", "likes", "Canada:ORG"))
        .addPrior(singleton(0.2, "Julie:PER", "origin", "Finnish:NAT"))
        .addPrior(singleton(1.0, "Arun:PER", "is", "Student:TIT"))
        .build();
    assertEquals(new HashSet<MLNText.Literal>() {{
      add(triple("Julie:PER", "likes", "Canada:ORG"));
      add(triple("Arun:PER", "is", "Student:TIT"));
    }}, net.gibbsMLE(1000).keySet());
  }

  @Test
  public void testBayesNetGibbsUnaryFactorsLarge() {
    Execution.threads = 1;
    // Setup
    int numClauses = 1000;
    int numIters = 100000;
    Iterator<String> stringIter1 = Utils.randomInsults(1);
    Iterator<String> stringIter2 = Utils.randomInsults(2);
    Random rand = new Random(42);

    // Add triples
    BayesNetBuilder builder = new BayesNetBuilder();
    builder.registerPredicate(new MLNText.Predicate("is an insult, like", "PERSON", "PERSON"));
    Set<MLNText.Literal> expectedPositive = new HashSet<>();

    for (int i = 0; i < numClauses; ++i) {
      double prob = rand.nextDouble();
      while (Math.abs(prob - 0.5) < 0.05) {
        prob = rand.nextDouble();
      }
      MLNText.Rule clause = singleton(prob, stringIter1.next() + ":PER", "is an insult, like", stringIter2.next() + ":PER");
      builder.addPrior(clause);
      if (prob > 0.5) {
        expectedPositive.add(clause.literals.get(0));
      }
    }

    // Inference (true sampling)
    Set<MLNText.Literal> inferredPositive = builder.build().gibbsMLE(numIters).keySet();
    Set<MLNText.Literal> overlap = CollectionUtils.intersection(expectedPositive, inferredPositive);
    assertFalse(overlap.equals(expectedPositive));
    // Got most expected positives right
    assertTrue(overlap.size() > expectedPositive.size() * 0.75);
    // Didn't get too many incorrect positives
    assertTrue(inferredPositive.size() < overlap.size() * 1.5);

    // Inference (with hillclimb)
    inferredPositive = builder.paramDoHillClimb(true).build().gibbsMAP(numIters).keySet();
    overlap = CollectionUtils.intersection(expectedPositive, inferredPositive);
    assertEquals(expectedPositive, overlap);
  }

  @Test
  public void testBayesNetChainRuleInference() {
    // Simple case
    Set<MLNText.Literal> inferredPositive;
    inferredPositive = new BayesNetBuilder()
            .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
            .registerPredicate(new MLNText.Predicate("origin", "PERSON", "COUNTRY"))
            .registerPredicate(new MLNText.Predicate("welcomes home", "COUNTRY", "PERSON"))
        .addRule(singleton(1.0, "Julie:PER", "origin", "Canada:CRY"))
        .addRule(binary(0.8, "Julie:PER", "origin", "Canada:CRY", "Julie:PER", "likes", "Canada:CRY"))
        .addRule(binary(0.8, "Julie:PER", "likes", "Canada:CRY", "Canada:CRY", "welcomes home", "Julie:PER"))
        .build().gibbsMAP(100).keySet();
    log(inferredPositive);
    assertTrue(inferredPositive.contains(triple("Canada:CRY", "welcomes home", "Julie:PER")));

    // Add priore
    inferredPositive = new BayesNetBuilder()
            .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
            .registerPredicate(new MLNText.Predicate("origin", "PERSON", "COUNTRY"))
            .registerPredicate(new MLNText.Predicate("welcomes home", "COUNTRY", "PERSON"))
        .addRule(singleton(0.29, "Julie:PER", "likes", "Canada:CRY"))
        .addRule(singleton(0.29, "Canada:CRY", "welcomes home", "Julie:PER"))
        .addRule(singleton(0.99, "Julie:PER", "origin", "Canada:CRY"))
        .addRule(binary(0.9, "Julie:PER", "origin", "Canada:CRY", "Julie:PER", "likes", "Canada:CRY"))
        .addRule(binary(0.9, "Julie:PER", "likes", "Canada:CRY", "Canada:CRY", "welcomes home", "Julie:PER"))
        .build().gibbsMAP(10000).keySet();
    log(inferredPositive);
    assertTrue(inferredPositive.contains(triple("Julie:PER", "likes", "Canada:CRY")));
    assertTrue(inferredPositive.contains(triple("Canada:CRY", "welcomes home", "Julie:PER")));

    // Lower weights, with priors
//    inferredPositive = new BayesNetBuilder()
//    .registerPredicate(new MLNText.Predicate("likes", "PERSON", "COUNTRY"))
//            .registerPredicate(new MLNText.Predicate("origin", "PERSON", "COUNTRY"))
//            .registerPredicate(new MLNText.Predicate("welcomes home", "COUNTRY", "PERSON"))
//            .addRule(singleton(0.01, "Julie:PER", "likes", "Canada:CRY"))
//            .addRule(singleton(0.01, "Canada:CRY", "welcomes home", "Julie:PER"))
//            .addRule(singleton(0.9, "Julie:PER", "origin", "Canada:CRY"))
//            .addRule(binary(0.6, "Julie:PER", "origin", "Canada:CRY", "Julie:PER", "likes", "Canada:CRY"))
//            .addRule(binary(0.6, "Julie:PER", "likes", "Canada:CRY", "Canada:CRY", "welcomes home", "Julie:PER"))
//            .build().gibbsMAP(10000);
//    log(inferredPositive);
//    assertTrue(inferredPositive.contains(triple("Julie:PER", "likes", "Canada:CRY")));
//    assertFalse(inferredPositive.contains(triple("Canada:CRY", "welcomes home", "Julie:PER")));
  }

//  /**
//   * A very simple test to ensure Tuffy can make interesting inferences.
//   */
//  @Test
//  public void testSimpleRuleInference() throws IOException {
//    MLNText basicMLNText = new MLNTextBuilder()
//            .addPredicates()
//            .openPredicate("LivesIn", "person", "place")
//            .openPredicate("WorksAt", "person", "org")
//            .openPredicate("HeadquarteredAt", "org", "place")
//            .endPredicates()
//            .addRules()
//            .newRule("unique place").orNot("HeadquarteredAt", "person", "place1" ).orNot("HeadquarteredAt", "person", "place2" ).equals("place1", "place2" ).endRule()
//            .endRules()
//            .end();
//
//    MLNText correlationRules = new MLNTextBuilder()
//            .addRules()
//            .newRule("location correlation").orNot("LivesIn", "person", "place").orNot("WorksAt", "person", "org").or("HeadquarteredAt", "org", "place").endRule(10)
//            .endRules()
//            .end();
//    MLNText evidences = new MLNTextBuilder()
//            .addRules()
//            .newRule().or("LivesIn", "Mark", "Oregon").endRule(10)
//            .newRule().or("WorksAt", "Mark", "IBM").endRule(10)
//            .newRule().or("HeadquarteredAt", "IBM", "Oregon").endRule(1)
//            .newRule().or("HeadquarteredAt", "IBM", "Seattle").endRule(10)
//            .endRules()
//            .end();
//
//    Tuffy result = new TuffyTextBuilder()
//            .addMLN("basic",basicMLNText)
//            .addMLN("evidence",evidences)
//            .query("LivesIn").query("WorksAt").query("HeadquarteredAt")
//            .infer();
//
//    assertTrue(result.getTruth("LivesIn", "Mark", "Oregon"));
//    assertTrue(result.getTruth("WorksAt", "Mark", "IBM"));
//    assertTrue(result.getTruth("HeadquarteredAt", "IBM", "Seattle"));
//    assertFalse(result.getTruth("HeadquarteredAt", "IBM", "Oregon"));
//
//    result = new TuffyTextBuilder()
//            .addMLN("basic",basicMLNText)
//            .addMLN("correlation",correlationRules)
//            .addMLN("evidence",evidences)
//            .query("LivesIn").query("WorksAt").query("HeadquarteredAt")
//            .infer();
//
//    assertTrue(result.getTruth("LivesIn", "Mark", "Oregon"));
//    assertTrue(result.getTruth("WorksAt", "Mark", "IBM"));
//    assertFalse(result.getTruth("HeadquarteredAt", "IBM", "Seattle"));
//    assertTrue(result.getTruth("HeadquarteredAt", "IBM", "Oregon"));
//  }

  @Test
  public void testTableFactor() {
    BayesNetBuilder.GroundedRule ruleA = new BayesNetBuilder.GroundedRule("0.2 A", Math.log(0.2), Math.log(0.8), 0);
    BayesNetBuilder.GroundedRule ruleBA = new BayesNetBuilder.GroundedRule("0.8 B => A", Math.log(0.8), Math.log(0.2), 0, 1);
    BayesNetBuilder.GroundedRule ruleCA = new BayesNetBuilder.GroundedRule("0.6 C => A", Math.log(0.6), Math.log(0.4), 0, 2);
    BayesNetBuilder.GroundedRule ruleCBA = new BayesNetBuilder.GroundedRule("0.4 C, B => A", Math.log(0.4), Math.log(0.6), 0, 1, 2);

    BayesNetBuilder.EagerTableFactor tf;
    // A
    tf = new BayesNetBuilder.EagerTableFactor(new ArrayList<>(Arrays.asList(ruleA)));
    Assert.assertEquals(tf.logProb(new boolean[]{false, false, false}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{false, false, true}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{false, true, false}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, false}), Math.log(0.2), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, true}), Math.log(0.2), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, false}), Math.log(0.2), 1e-5);

    // B => A
    tf = new BayesNetBuilder.EagerTableFactor(new ArrayList<>(Arrays.asList(ruleBA, ruleA)));
    Assert.assertEquals(tf.logProb(new boolean[]{false, false, false}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, false}), Math.log(0.2), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, false, true}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, true}), Math.log(0.2), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, true, false}), Math.log(0.2), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, false}), Math.log(0.8), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, true, true}), Math.log(0.2), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, true}), Math.log(0.8), 1e-5);

    // C => A
    tf = new BayesNetBuilder.EagerTableFactor(new ArrayList<>(Arrays.asList(ruleCA, ruleA)));
    Assert.assertEquals(tf.logProb(new boolean[]{false, false, false}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, false}), Math.log(0.2), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, false, true}), Math.log(0.4), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, true}), Math.log(0.6), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, true, false}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, false}), Math.log(0.2), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, true, true}), Math.log(0.4), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, true}), Math.log(0.6), 1e-5);

    // B, C => A
    tf = new BayesNetBuilder.EagerTableFactor(new ArrayList<>(Arrays.asList(ruleCBA, ruleCA, ruleBA, ruleA)));
    Assert.assertEquals(tf.logProb(new boolean[]{false, false, false}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, false}), Math.log(0.2), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, false, true}), Math.log(0.4), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, true}), Math.log(0.6), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, true, false}), Math.log(0.2), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, false}), Math.log(0.8), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, true, true}), Math.log(0.6), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, true}), Math.log(0.4), 1e-5);

    // C => A, B => A
    tf = new BayesNetBuilder.EagerTableFactor(new ArrayList<>(Arrays.asList(ruleCA, ruleBA, ruleA)));
    Assert.assertEquals(tf.logProb(new boolean[]{false, false, false}), Math.log(0.8), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, false}), Math.log(0.2), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, false, true}), Math.log(0.4), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, false, true}), Math.log(0.6), 1e-5);

    Assert.assertEquals(tf.logProb(new boolean[]{false, true, false}), Math.log(0.2), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, false}), Math.log(0.8), 1e-5);

    double prob = (Math.log(0.8) + Math.log(0.6))/2.0;
    Assert.assertEquals(tf.logProb(new boolean[]{false, true, true}), Math.log(1.0 - Math.exp(prob)), 1e-5);
    Assert.assertEquals(tf.logProb(new boolean[]{true, true, true}), prob, 1e-5);

  }


}
