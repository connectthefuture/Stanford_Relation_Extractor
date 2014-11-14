package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.entitylinking.EntityLinker;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.util.MetaClass;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.Assert.*;

/**
 * A test for the graph consistency post processors.
 *
 * @author Gabor Angeli
 */
public class GraphConsistencyPostProcessorsTest extends PostProcessorsData {

  private static EntityGraph asSimpleGraph(Map<KBPEntity, List<KBPSlotFill>> fills) {
    EntityGraph graph = new EntityGraph();
    for (Map.Entry<KBPEntity, List<KBPSlotFill>> entry : fills.entrySet()) {
      for (KBPSlotFill fill : entry.getValue()) {
        graph.add(entry.getKey(), fill.key.getSlotEntity().orCrash(), fill);
      }
    }
    return graph;
  }

  @Before
  public void initEntityLinker() {
    Props.ENTITYLINKING_LINKER = Lazy.from((EntityLinker) MetaClass.create(Props.KBP_ENTITYLINKER_CLASS).createInstance());
  }

  private boolean graph_altnames = Props.TEST_GRAPH_ALTNAMES_DO;
  @Before
  public void tweakOptions() {
    Props.TEST_GRAPH_ALTNAMES_DO = true;
  }

  @After
  public void restoreOptions() {
    Props.TEST_GRAPH_ALTNAMES_DO = graph_altnames;
  }


  /**
   * Ensure we filter the same things in GraphConsistency as we do in {@link edu.stanford.nlp.kbp.slotfilling.evaluate.HeuristicSlotfillPostProcessorsTest#testNoDuplicatesApproximate()}.
   */
  @Test
  public void testMergeNoDuplicatesApproximateJulie() {
    Map<KBPEntity, List<KBPSlotFill>> filtered
        = new GraphConsistencyPostProcessors.EntityMergingPostProcessor().postProcess(asSimpleGraph(approximateDuplicateData())).toMap();

    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_SIBLINGS, "Adan Chavez", 1.0)));
    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_SIBLINGS, "Adan", 0.9)) );  // can't say these are the same anymore!
    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "singer/songwriter", 1.0)));
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "Singer\\/songwriter", 0.9)));
    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "Socialist Party", 1.0)));
    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "United Socialist party", 0.9)));  // should no longer say these are the same
    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "murder defendant", 1.0)));
    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "defendant", 0.9)));  // should no longer say these are the same
    assertTrue(  filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "American Family Association", 1.0)) );
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "AFA", 0.9)) );
  }

  /**
   * Ensure we filter the same things in GraphConsistency as we do in {@link edu.stanford.nlp.kbp.slotfilling.evaluate.HeuristicSlotfillPostProcessorsTest#testNoDuplicatesApproximate()}..
   * The exception to this is alternate names, which we only group here, and take the transitive completion of in
   * {@link GraphConsistencyPostProcessors.TransitiveRelationPostProcessor}, tested in
   * {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GraphConsistencyPostProcessorsTest#testTransitiveCompletion()}.
   */
  @Test
  public void testMergeNoDuplicatesApproximateStanford() {
    Map<KBPEntity, List<KBPSlotFill>> filtered
        = new GraphConsistencyPostProcessors.EntityMergingPostProcessor().postProcess(asSimpleGraph(approximateDuplicateData())).toMap();

    assertTrue(  filtered.get(stanford).contains(fill(stanford, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, "California", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, "california", 0.9)) );
    assertTrue(  filtered.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl Â Blake", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl Blake", 0.9)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl  Blake", 0.9)) );
    assertTrue(  filtered.get(stanford).contains(fill(stanford, RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, "60,000", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, "\"60,000\"", 0.9)) );
  }

  /**
   * Test that we take the transitive completion of alternate names for the test entities tested in
   * {@link edu.stanford.nlp.kbp.slotfilling.evaluate.HeuristicSlotfillPostProcessorsTest#testNoDuplicatesApproximate()}.
   */
  @Test
  public void testTransitiveCompletion() {
    // Base Case 1: Only Approximate Duplicates
    Map<KBPEntity, List<KBPSlotFill>> filteredSimple
        = new GraphConsistencyPostProcessors.EntityMergingPostProcessor().postProcess(asSimpleGraph(approximateDuplicateData())).toMap();
    assertTrue(  filteredSimple.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities", 1.0)) );
    assertFalse( filteredSimple.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities LLC", 0.9)) );
    assertFalse( filteredSimple.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard Madoff Investment Securities", 0.9)) );
    assertTrue(  filteredSimple.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Illinois Tool Works , Inc.", 1.0)) );
    assertFalse( filteredSimple.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "ITW", 0.8)) );

    // Base Case 2: Only Transitive
    Map<KBPEntity, List<KBPSlotFill>> filteredTrans
        = new GraphConsistencyPostProcessors.TransitiveRelationPostProcessor().postProcess(asSimpleGraph(approximateDuplicateData())).toMap();
    assertTrue( filteredTrans.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities", 1.0)) );
    assertTrue( filteredTrans.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities LLC", 0.9)) );
    assertTrue( filteredTrans.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard Madoff Investment Securities", 0.9)) );
    assertTrue( filteredTrans.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Illinois Tool Works , Inc.", 1.0)) );
    assertTrue( filteredTrans.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "ITW", 0.8)) );

    // Test: Approximate + Transitive
    Map<KBPEntity, List<KBPSlotFill>> filtered
        = GraphConsistencyPostProcessor.all(
            new GraphConsistencyPostProcessors.EntityMergingPostProcessor(),
            new GraphConsistencyPostProcessors.TransitiveRelationPostProcessor()
          ).postProcess(asSimpleGraph(approximateDuplicateData())).toMap();
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities LLC", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard Madoff Investment Securities", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Illinois Tool Works , Inc.", 1.0)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "ITW", 0.8)) );
  }

  /**
   * A very simple test to see if we can symmeterize relations.
   */
  @Test
  public void testSymmetricRelations() {
    // Baseline: simply convert the known fill to a graph
    Map<KBPEntity, List<KBPSlotFill>> baseline = asSimpleGraph(reflexiveData()).toMap();
    assertTrue( baseline.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Julie", NERTag.PERSON, 0.8)) );
    assertFalse(  baseline.get(julie).contains(fill(julie, RelationType.PER_EMPLOYEE_OF, "Stanford", NERTag.ORGANIZATION, 0.8)) );
    assertFalse( baseline.get(julie).contains(fill(julie, RelationType.PER_EMPLOYEE_OF, "Stanford University", NERTag.ORGANIZATION, 0.8)) );

    // Test: symmeterize the graph
    Map<KBPEntity, List<KBPSlotFill>> filtered
        = new GraphConsistencyPostProcessors.SymmetricFunctionRewritePostProcessor().postProcess(asSimpleGraph(reflexiveData())).toMap();
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Julie", NERTag.PERSON, 0.8)) );
    assertTrue( filtered.get(julie).contains(fill(julie, RelationType.PER_EMPLOYEE_OF, "Stanford", NERTag.ORGANIZATION, 0.8)) );
    // TODO(gabor) this should really be a true inference as well -- if A rel B, and B altname C, then A rel C.
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_EMPLOYEE_OF, "Stanford University", NERTag.ORGANIZATION, 0.8)) );
  }

  @Test
  public void testEntityMergingRegression1() {
    Lazy<EntityLinker> oldLinker = Props.ENTITYLINKING_LINKER;
    Props.ENTITYLINKING_LINKER = Lazy.<EntityLinker>from(new EntityLinker.GaborsHighPrecisionBaseline());

    // Create data
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();
    data.put(julie, new ArrayList<KBPSlotFill>());
    data.put(hamilton, new ArrayList<KBPSlotFill>());
    data.get(julie).add(fill(julie, RelationType.PER_TITLE, "Student", NERTag.TITLE, 0.6));
    data.get(julie).add(fill(julie, RelationType.PER_EMPLOYEE_OF, "Stanford", NERTag.ORGANIZATION, 0.6));
    data.get(hamilton).add(fill(hamilton, RelationType.PER_TITLE, "Student", NERTag.TITLE, 0.6));
    data.get(hamilton).add(fill(hamilton, RelationType.PER_EMPLOYEE_OF, "Stanford", NERTag.ORGANIZATION, 0.6));

    Map<KBPEntity, List<KBPSlotFill>> filtered
        = new GraphConsistencyPostProcessors.EntityMergingPostProcessor().postProcess(asSimpleGraph(data)).toMap();
    assertEquals(0, filtered.get(julie).size());
    assertTrue( filtered.get(hamilton).contains(fill(hamilton, RelationType.PER_TITLE, "Student", NERTag.TITLE, 0.6)) );
    assertTrue( filtered.get(hamilton).contains(fill(hamilton, RelationType.PER_EMPLOYEE_OF, "Stanford", NERTag.ORGANIZATION, 0.6)) );

    Props.ENTITYLINKING_LINKER = oldLinker;
  }
}
