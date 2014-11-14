package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.EntityGraph;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.HashSet;
import java.util.List;

import static org.junit.Assert.*;

/**
 * A test for {@link SimpleGraphInferenceEngine}.
 * Much of this is adapted from the old graph inference engine test,
 * created by Arun.
 *
 * @author Gabor Angeli
 */
public class SimpleGraphInferenceEngineTest {

  private static void addEdge(EntityGraph graph, KBPEntity head, KBPEntity tail, String relation, NERTag type ) {
    graph.add( head, tail, KBPNew.from(head).slotValue(tail.name).slotType(type).rel(relation).KBPSlotFill() );
  }

  private static EntityGraph mkGraph() {
    EntityGraph orgGraph = new EntityGraph();
    KBPEntity stanford = KBPNew.entName("Stanford Univ.").entType(NERTag.ORGANIZATION).KBPEntity();
    orgGraph.addVertex(stanford);

    // Stanford Univ. and Stanford CSE are alternate names
    KBPEntity cseStanford = KBPNew.entName("Stanford CSE").entType(NERTag.ORGANIZATION).KBPEntity();
    orgGraph.addVertex(cseStanford);
    addEdge(orgGraph, stanford, cseStanford, RelationType.ORG_ALTERNATE_NAMES.canonicalName, NERTag.ORGANIZATION);

    // What would stanford be without Julie?
    KBPEntity julie = KBPNew.entName("Julie").entType(NERTag.PERSON).KBPEntity();
    orgGraph.addVertex(julie);
    addEdge(orgGraph, stanford, julie, "org:top_members/employees", NERTag.PERSON);
    addEdge(orgGraph, julie, stanford, "per:member_of", NERTag.ORGANIZATION);

    // Let's add Gabor who's highly ranked despite not being part of Stanford, it seems
    KBPEntity gabor = KBPNew.entName("Gabor").entType(NERTag.PERSON).KBPEntity();
    orgGraph.addVertex(gabor);
    addEdge(orgGraph, stanford, gabor, "org:top_members/employees", NERTag.PERSON);

    // Add multiple rules that should match
    KBPEntity year1885 = KBPNew.entName("1885").entType(NERTag.DATE).KBPEntity();
    KBPEntity lastYear = KBPNew.entName("2012").entType(NERTag.DATE).KBPEntity();
    KBPEntity california = KBPNew.entName("California").entType(NERTag.STATE_OR_PROVINCE).KBPEntity();
    KBPEntity stanfordCity = KBPNew.entName("Stanford").entType(NERTag.ORGANIZATION).KBPEntity();
    orgGraph.addVertex(year1885);
    orgGraph.addVertex(stanfordCity);
    addEdge(orgGraph, stanford, stanfordCity, RelationType.ORG_ALTERNATE_NAMES.canonicalName, NERTag.ORGANIZATION);
    addEdge(orgGraph, stanford, year1885, "found in", NERTag.DATE);
    addEdge(orgGraph, stanford, lastYear, "last year", NERTag.DATE);
    addEdge(orgGraph, stanford, california, "org:stateorprovince_of_headquarters", NERTag.STATE_OR_PROVINCE);

    // Asserts
    assertEquals(1, orgGraph.getEdges(stanford, cseStanford).size());
    assertEquals(1, orgGraph.getEdges(stanford, julie).size());
    assertEquals(1, orgGraph.getEdges(julie, stanford).size());
    assertEquals(1, orgGraph.getEdges(stanford, gabor).size());
    assertEquals(1, orgGraph.getEdges(stanford, stanfordCity).size());
    assertEquals(1, orgGraph.getEdges(stanford, year1885).size());
    assertEquals(1, orgGraph.getEdges(stanford, lastYear).size());
    assertEquals(1, orgGraph.getEdges(stanford, california).size());
    assertEquals(7, orgGraph.getOutDegree(stanford));

    return orgGraph;
  }

  private static SimpleGraphInferenceEngine mkInferenceEngine() {
    return new SimpleGraphInferenceEngine(new BufferedReader(new StringReader(
        "found_in(ORGANIZATION,DATE)\n" +
        "org:founded(ORGANIZATION,DATE)\n" +
        "org:alternate_names(ORGANIZATION,ORGANIZATION)\n" +
        "per:member_of(PERSON,ORGANIZATION)\n" +
        "org:stateorprovince_of_headquarters(ORGANIZATION,STATE_OR_PROVINCE)\n" +
        "per:stateorprovinces_of_residence(PERSON,STATE_OR_PROVINCE)\n" +
        "last_year(ORGANIZATION,DATE)\n" +
        "2.3  !found_in(x0,x1)vorg:founded(x0,x1)\n" +
        "5.0  !found_in(x0,x1)v!org:alternate_names(x0,x2)vorg:founded(x2,x1)\n" +
        "-1.2  !last_year(x0,x1)vorg:founded(x0,x1)\n" +
        "10.0  !per:member_of(x0,x1)v!org:stateorprovince_of_headquarters(x1,x2)vper:stateorprovinces_of_residence(x0,x2)\n"
    )), 0.0);
  }

  @Test
  public void testMkGraph() {
    mkGraph();
  }

  @Test
  public void testParseRules() {
    assertEquals(2, mkInferenceEngine().antecedentsForRelation.get(RelationType.ORG_FOUNDED).size());
    assertEquals(1, mkInferenceEngine().antecedentsForRelation.get(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE).size());
  }

  @Test
  public void testBoundRuleBasicCase() {
    EntityGraph graph = mkGraph();
    SimpleGraphInferenceEngine.Rule rule = mkInferenceEngine().antecedentsForRelation.get(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE).iterator().next();
    SimpleGraphInferenceEngine.BoundRule boundRule = rule.bindConsequent(KBPNew.entName("Julie").entType(NERTag.PERSON).slotValue("California").slotType(NERTag.STATE_OR_PROVINCE).KBPair());
    // Julie works for Stanford Univ.
    assertTrue(boundRule.isConsistent(graph, "x1", KBPNew.entName("Stanford Univ.").entType(NERTag.ORGANIZATION).KBPEntity()));
    // Julie is not registered to work for Stanford
    assertFalse(boundRule.isConsistent(graph, "x1", KBPNew.entName("Stanford").entType(NERTag.ORGANIZATION).KBPEntity()));
    // X1 is the only free variable
    assertEquals(new HashSet<String>(){{add("x1");}}, boundRule.freeVariables());
    // Stanford Univ. is the only valid assignment
    for (KBPEntity candidate : graph.getAllVertices()) {
      if (!candidate.equals(KBPNew.entName("Stanford Univ.").entType(NERTag.ORGANIZATION).KBPEntity())) {
        assertFalse(boundRule.isConsistent(graph, "x1", candidate));
      } else {
        assertTrue(boundRule.isConsistent(graph, "x1", candidate));
      }
    }
  }

  @Test
  public void testApplyBasicCaseJulie() {
    // Get variables and data
    KBPEntity julie = KBPNew.entName("Julie").entType(NERTag.PERSON).KBPEntity();
    EntityGraph graph = mkGraph();
    assertEquals(1, graph.getOutDegree(julie));
    SimpleGraphInferenceEngine engine = mkInferenceEngine();
    // Run inference
    EntityGraph modified = engine.apply(graph, julie);
    assertEquals(2, modified.getOutDegree(julie));
    assertTrue(modified.getEdges(julie, KBPNew.entName("California").entType(NERTag.STATE_OR_PROVINCE).KBPEntity()).contains(
        KBPNew.from(julie).slotValue("California").slotType(NERTag.STATE_OR_PROVINCE).rel(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE.canonicalName).KBPSlotFill()));
  }

  @Test
  public void testApplyBasicCaseStanford() {
    // Get variables and data
    KBPEntity stanford = KBPNew.entName("Stanford Univ.").entType(NERTag.ORGANIZATION).KBPEntity();
    EntityGraph graph = mkGraph();
    SimpleGraphInferenceEngine engine = mkInferenceEngine();
    // Run inference
    EntityGraph modified = engine.apply(graph, stanford);
    KBPSlotFill fill = KBPNew.from(stanford).slotValue("1885").slotType(NERTag.DATE).rel(RelationType.ORG_FOUNDED.canonicalName).KBPSlotFill();
    List<KBPSlotFill> edges = modified.getEdges(stanford, KBPNew.entName("1885").entType(NERTag.DATE).KBPEntity());
    assertTrue(edges.contains(fill));
  }

  @Test
  public void testApplyBasicCaseStanfordCity() {
    // Get variables and data
    KBPEntity stanfordCity = KBPNew.entName("Stanford").entType(NERTag.ORGANIZATION).KBPEntity();
    EntityGraph graph = mkGraph();
    SimpleGraphInferenceEngine engine = mkInferenceEngine();
    // Run inference
    EntityGraph modified = engine.apply(graph, stanfordCity);
    modified = engine.apply(modified, stanfordCity);
    assertTrue(modified.getEdges(stanfordCity, KBPNew.entName("1885").entType(NERTag.DATE).KBPEntity()).contains(
        KBPNew.from(stanfordCity).slotValue("1885").slotType(NERTag.DATE).rel(RelationType.ORG_FOUNDED.canonicalName).KBPSlotFill()));

  }

}
