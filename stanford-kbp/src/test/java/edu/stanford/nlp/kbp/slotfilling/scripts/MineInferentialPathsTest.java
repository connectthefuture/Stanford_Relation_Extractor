package edu.stanford.nlp.kbp.slotfilling.scripts;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.EntityGraph;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import org.junit.Test;

import java.util.*;
import java.util.stream.Stream;

import static junit.framework.Assert.*;

/**
 * I'm convinced enough that I'm doing something wrong that I think it deserves a test.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public class MineInferentialPathsTest {

  private KBPEntity julie = KBPNew.entName("Julie").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity arun = KBPNew.entName("Arun").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity gabor = KBPNew.entName("Gabor").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity chris = KBPNew.entName("Chris").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity percy = KBPNew.entName("Percy").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity stanford = KBPNew.entName("Stanford").entType(NERTag.ORGANIZATION).KBPEntity();
  private KBPEntity canada = KBPNew.entName("Canada").entType(NERTag.COUNTRY).KBPEntity();
  private KBPEntity india = KBPNew.entName("India").entType(NERTag.COUNTRY).KBPEntity();

  private KBPEntity px0 = KBPNew.entName("x0").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity px1 = KBPNew.entName("x1").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity px2 = KBPNew.entName("x2").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity px3 = KBPNew.entName("x3").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity px4 = KBPNew.entName("x4").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity ox0 = KBPNew.entName("x0").entType(NERTag.ORGANIZATION).KBPEntity();
  private KBPEntity ox1 = KBPNew.entName("x1").entType(NERTag.ORGANIZATION).KBPEntity();
  private KBPEntity ox2 = KBPNew.entName("x2").entType(NERTag.ORGANIZATION).KBPEntity();
  private KBPEntity ox3 = KBPNew.entName("x3").entType(NERTag.ORGANIZATION).KBPEntity();
  private KBPEntity cx0 = KBPNew.entName("x0").entType(NERTag.COUNTRY).KBPEntity();
  private KBPEntity cx1 = KBPNew.entName("x1").entType(NERTag.COUNTRY).KBPEntity();

  @SuppressWarnings("unchecked")
  KBPIR dummyIR = new KBPIR() {
    @Override
    protected <E extends CoreMap> List<E> queryCoreMaps(String tableName, Class<E> expectedOutput, KBPEntity entity, Maybe<KBPEntity> slotValue, Maybe<String> relation, Set<String> docidsToForce, int maxDocuments, boolean officialIndexOnly) {
      return Collections.EMPTY_LIST;
    }
    @Override
    public Annotation fetchDocument(String docId, boolean officialIndexOnly) {
      return new Annotation("");
    }
    @Override
    public int queryNumHits(Collection<String> terms) {
      return 1;
    }

    @Override
    public Stream<Annotation> slurpDocuments(int maxDocuments) {
      return Stream.empty();
    }

    @Override
    protected List<String> queryDocIDs(String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotValue, Maybe<NERTag> slotValueType, int maxDocuments, boolean officialIndexOnly) {
      return Collections.EMPTY_LIST;
    }
    @Override
    public Set<String> getKnownRelationsForPair(KBPair pair) {
      return Collections.EMPTY_SET;
    }
    @Override
    public List<KBPSlotFill> getKnownSlotFillsForEntity(KBPEntity entity) {
      return Collections.EMPTY_LIST;
    }

    @Override
    public IterableIterator<Pair<Annotation, Double>> queryKeywords(Collection<String> words, Maybe<Integer> maxDocs) {
      return new IterableIterator<Pair<Annotation, Double>>(Collections.EMPTY_LIST.iterator());
    }
  };

  @Test
  public void testFlatPath() {
    EntityGraph graph = new EntityGraph();
    graph.add(julie, canada, KBPNew.from(julie).slotValue(canada).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBPSlotFill());
    graph.add(canada, gabor, KBPNew.from(canada).slotValue(gabor).rel(RelationType.ORG_FOUNDED_BY).KBPSlotFill());
    graph.add(gabor, stanford, KBPNew.from(gabor).slotValue(stanford).rel(RelationType.PER_EMPLOYEE_OF).KBPSlotFill());

    Counter<List<KBTriple>> counts = MineInferentialPaths.extractAllFormulas(graph, dummyIR);

    assertEquals(6, counts.size());
    assertEquals(6.0, counts.totalCount(), 1e-5);
    assertEquals(1.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(cx1).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple())));
    assertEquals(1.0, counts.getCount(Collections.singletonList(KBPNew.from(cx0).slotValue(px1).rel(RelationType.ORG_FOUNDED_BY).KBTriple())));
    assertEquals(1.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple())));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(cx0).slotValue(px1).rel(RelationType.ORG_FOUNDED_BY).KBTriple());
      add(KBPNew.from(px2).slotValue(cx0).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple());
    }}));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(cx0).slotValue(px1).rel(RelationType.ORG_FOUNDED_BY).KBTriple());
      add(KBPNew.from(px1).slotValue(ox2).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
    }}));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(cx0).slotValue(px1).rel(RelationType.ORG_FOUNDED_BY).KBTriple());
      add(KBPNew.from(px2).slotValue(cx0).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple());
      add(KBPNew.from(px1).slotValue(ox3).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
    }}));
  }

  @Test
  public void testLengthOneEntailments() {
    EntityGraph graph = new EntityGraph();
    graph.add(julie, canada, KBPNew.from(julie).slotValue(canada).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBPSlotFill());
    graph.add(arun, india, KBPNew.from(arun).slotValue(india).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBPSlotFill());
    graph.add(gabor, stanford, KBPNew.from(gabor).slotValue(stanford).rel(RelationType.PER_EMPLOYEE_OF).KBPSlotFill());

    Counter<List<KBTriple>> counts = MineInferentialPaths.extractAllFormulas(graph, dummyIR);

    assertEquals(2, counts.size());
    assertEquals(3.0, counts.totalCount(), 1e-5);
    assertEquals(2.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(cx1).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple())));
    assertEquals(1.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple())));
  }

  @Test
  public void testLengthTwoEntailmentsReverbTranslations() {
    EntityGraph graph = new EntityGraph();
    graph.add(julie, canada, KBPNew.from(julie).slotValue(canada).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBPSlotFill());
    graph.add(julie, canada, KBPNew.from(julie).slotValue(canada).rel("born_in").KBPSlotFill());
    graph.add(julie, canada, KBPNew.from(julie).slotValue(canada).rel("birthed_in").KBPSlotFill());
    graph.add(arun, india, KBPNew.from(arun).slotValue(india).rel("born_in").KBPSlotFill());
    graph.add(arun, india, KBPNew.from(arun).slotValue(india).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBPSlotFill());

    Counter<List<KBTriple>> counts = MineInferentialPaths.extractAllFormulas(graph, dummyIR);

    assertEquals(9, counts.size());
    assertEquals(13.0, counts.totalCount(), 1e-5);
    // singletons
    assertEquals(2.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(cx1).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple())));
    assertEquals(2.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(cx1).rel("born_in").KBTriple())));
    assertEquals(1.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(cx1).rel("birthed_in").KBTriple())));
    // pairs
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(cx1).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple());
      add(KBPNew.from(px0).slotValue(cx1).rel("born_in").KBTriple());
    }}));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(cx1).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple());
      add(KBPNew.from(px0).slotValue(cx1).rel("birthed_in").KBTriple());
    }}));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(cx1).rel("born_in").KBTriple());
      add(KBPNew.from(px0).slotValue(cx1).rel("birthed_in").KBTriple());
    }}));
    // reverse entailments
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(cx1).rel("born_in").KBTriple());
      add(KBPNew.from(px0).slotValue(cx1).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple());
    }}));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(cx1).rel("birthed_in").KBTriple());
      add(KBPNew.from(px0).slotValue(cx1).rel(RelationType.PER_COUNTRY_OF_BIRTH).KBTriple());
    }}));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(cx1).rel("birthed_in").KBTriple());
      add(KBPNew.from(px0).slotValue(cx1).rel("born_in").KBTriple());
    }}));
  }

  @Test
  public void testLengthThreeEntailments() {
    EntityGraph graph = new EntityGraph();
    graph.add(julie, stanford, KBPNew.from(julie).slotValue(stanford).rel(RelationType.PER_EMPLOYEE_OF).KBPSlotFill());
    graph.add(arun, stanford, KBPNew.from(arun).slotValue(stanford).rel(RelationType.PER_EMPLOYEE_OF).KBPSlotFill());
    graph.add(arun, julie, KBPNew.from(arun).slotValue(julie).rel(RelationType.PER_SIBLINGS).KBPSlotFill());
    graph.add(julie, arun, KBPNew.from(julie).slotValue(arun).rel(RelationType.PER_SIBLINGS).KBPSlotFill());

    Counter<List<KBTriple>> counts = MineInferentialPaths.extractAllFormulas(graph, dummyIR);

    // singletons
    assertEquals(2.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple())));
    assertEquals(2.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(px1).rel(RelationType.PER_SIBLINGS).KBTriple())));
    // some pairs
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px2).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
    }}));
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px0).slotValue(px2).rel(RelationType.PER_SIBLINGS).KBTriple());
    }}));
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px2).slotValue(px0).rel(RelationType.PER_SIBLINGS).KBTriple());
    }}));
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(px1).rel(RelationType.PER_SIBLINGS).KBTriple());
      add(KBPNew.from(px1).slotValue(px0).rel(RelationType.PER_SIBLINGS).KBTriple());
    }}));
    // triples
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px2).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px0).slotValue(px2).rel(RelationType.PER_SIBLINGS).KBTriple());
    }}));
    assertEquals(1.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px2).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px2).slotValue(px0).rel(RelationType.PER_SIBLINGS).KBTriple());
    }}));
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px2).slotValue(px0).rel(RelationType.PER_SIBLINGS).KBTriple());
      add(KBPNew.from(px2).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
    }}));
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
      add(KBPNew.from(px0).slotValue(px2).rel(RelationType.PER_SIBLINGS).KBTriple());
      add(KBPNew.from(px2).slotValue(ox1).rel(RelationType.PER_EMPLOYEE_OF).KBTriple());
    }}));

    assertEquals(10, counts.size());  // did we get them all?
  }

  @Test
  public void testLoop2() {
    EntityGraph graph = new EntityGraph();
    graph.add(julie, arun, KBPNew.from(julie).slotValue(arun).rel("1").KBPSlotFill());
    graph.add(arun, julie, KBPNew.from(arun).slotValue(julie).rel("1").KBPSlotFill());

    Counter<List<KBTriple>> counts = MineInferentialPaths.extractAllFormulas(graph, dummyIR);

    assertEquals(2.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple())));
    assertEquals(2.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple());
      add(KBPNew.from(px1).slotValue(px0).rel("1").KBTriple());
    }}));
  }

  @Test
  public void testLoop3() {
    EntityGraph graph = new EntityGraph();
    graph.add(julie, arun, KBPNew.from(julie).slotValue(arun).rel("1").KBPSlotFill());
    graph.add(arun, gabor, KBPNew.from(arun).slotValue(gabor).rel("1").KBPSlotFill());
    graph.add(gabor, julie, KBPNew.from(gabor).slotValue(julie).rel("1").KBPSlotFill());

    Counter<List<KBTriple>> counts = MineInferentialPaths.extractAllFormulas(graph, dummyIR);

    assertEquals(3.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple())));
    assertEquals(3.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple());
      add(KBPNew.from(px1).slotValue(px2).rel("1").KBTriple());
    }}));
    assertEquals(3.0, counts.getCount(new ArrayList<KBTriple>() {{
      add(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple());
      add(KBPNew.from(px1).slotValue(px2).rel("1").KBTriple());
      add(KBPNew.from(px2).slotValue(px0).rel("1").KBTriple());
    }}));
  }

  @Test
  public void testComplexGraph() {
    EntityGraph graph = new EntityGraph();
    graph.add(julie, arun, KBPNew.from(julie).slotValue(arun).rel("1").KBPSlotFill());
    graph.add(arun, gabor, KBPNew.from(arun).slotValue(gabor).rel("1").KBPSlotFill());
    graph.add(gabor, julie, KBPNew.from(gabor).slotValue(julie).rel("2").KBPSlotFill());
    graph.add(arun, chris, KBPNew.from(arun).slotValue(chris).rel("1").KBPSlotFill());
    graph.add(chris, julie, KBPNew.from(chris).slotValue(julie).rel("1").KBPSlotFill());
    graph.add(chris, julie, KBPNew.from(chris).slotValue(julie).rel("2").KBPSlotFill());
    graph.add(chris, percy, KBPNew.from(chris).slotValue(percy).rel("1").KBPSlotFill());

    Counter<List<KBTriple>> counts = MineInferentialPaths.extractAllFormulas(graph, dummyIR);


    // Sanity Checks
    assertEquals(5.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple())));
    assertEquals(2.0, counts.getCount(Collections.singletonList(KBPNew.from(px0).slotValue(px1).rel("2").KBTriple())));

    // Some Probabilities
    // P(a1b ^ b2c -> c1a) = 2 / 2
    assertEquals(1.0,
        counts.getCount(new ArrayList<KBTriple>() {{
          add(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple());
          add(KBPNew.from(px1).slotValue(px2).rel("2").KBTriple());
          add(KBPNew.from(px2).slotValue(px0).rel("1").KBTriple());
        }}) /
        counts.getCount(new ArrayList<KBTriple>() {{
          add(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple());
          add(KBPNew.from(px1).slotValue(px2).rel("2").KBTriple());
        }}),
        1e-5);

    // P(a1b ^ b1c -> c2a) = 2 / 5
    assertEquals(0.4,
        counts.getCount(new ArrayList<KBTriple>() {{
          add(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple());
          add(KBPNew.from(px1).slotValue(px2).rel("1").KBTriple());
          add(KBPNew.from(px2).slotValue(px0).rel("2").KBTriple());
        }}) /
        counts.getCount(new ArrayList<KBTriple>() {{
          add(KBPNew.from(px0).slotValue(px1).rel("1").KBTriple());
          add(KBPNew.from(px1).slotValue(px2).rel("1").KBTriple());
        }}),
    1e-5);
  }
}
