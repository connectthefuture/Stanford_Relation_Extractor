package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.graph.DirectedMultiGraph;
import java.util.function.Function;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import junit.framework.Assert;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Tests methods in CollectionsUtils
 */
public class CollectionUtilsTest {

  @Test
  public void testFilterList() {
    List<Integer> input = new ArrayList<Integer>(){{ add(1); add(2); add(3); add(4); add(5); }};
    assertEquals(new ArrayList<Integer>(){{ add(1); add(3); add(5); }},
      CollectionUtils.filter(input, in -> in % 2 == 1)
    );
  }

  @Test
  public void testFilterIterator() {
    Iterator<Integer> input = new ArrayList<Integer>(){{ add(1); add(2); add(3); add(4); add(5); }}.iterator();
    Iterator<Integer> output = CollectionUtils.filter(input, in -> in % 2 == 1);
    assertEquals(new Integer(1), output.next());
    assertEquals(new Integer(3), output.next());
    assertEquals(new Integer(5), output.next());
    assertFalse(output.hasNext());
  }

  // mergeVertices
  @Test
  public void testMergeVertices() {
    CollectionUtils.EdgeRewriter<String,String> edgeRewriter = new CollectionUtils.EdgeRewriter<String, String>() {
      @Override
      public boolean sameEdge(String edge1, String edge2) {
        String[] parts1 = edge1.split(" -> ");
        String[] parts2 = edge1.split(" -> ");
        return ( parts1[0].equals( parts2[0]) || parts1[1].equals(parts2[1]));
      }

      @Override
      public boolean isValidOutgoingEdge(String pivot, String fill) {
        return true;
      }

      @Override
      public String mergeEdges(String edge1, String edge2) {
        return edge1;
      }

      @Override
      public String rewrite(String pivot, String newValue, String edge) {
        String[] parts = edge.split(" -> ");
        assert parts.length == 2;
        if( parts[0].equals(pivot))
          return pivot + " -> " + newValue;
        else
          return newValue + " -> " + pivot;
      }
    };

    // Null child
    DirectedMultiGraph<String, String> graph1 = new DirectedMultiGraph<String,String>() {{
      addVertex("A");
      add("B", "C", "B -> C");
    }};
    // Assertions
    CollectionUtils.mergeVertices(graph1, "A", "B", edgeRewriter);
    Assert.assertEquals(new LinkedHashSet<String>() {{
        add("A");
        add("C");
      }},
      graph1.getAllVertices());
    Assert.assertEquals(new LinkedHashSet<String>() {{ add("C"); }},
            graph1.getChildren("A") );
    Assert.assertEquals(new ArrayList<String>() {{ add("A -> C"); }},
            graph1.getEdges("A","C"));

    // Null parent
    DirectedMultiGraph<String, String> graph2 = new DirectedMultiGraph<String,String>() {{
      addVertex("A");
      add("B", "C", "B -> C");
    }};
    // Assertions
    CollectionUtils.mergeVertices(graph2, "A", "C", edgeRewriter);
    Assert.assertEquals(new LinkedHashSet<String>() {{
        add("A");
        add("B");
      }},
      graph2.getAllVertices());
    Assert.assertEquals(new LinkedHashSet<String>() {{ add("A"); }},
            graph2.getChildren("B") );
    Assert.assertEquals(new ArrayList<String>() {{ add("B -> A"); }},
            graph2.getEdges("B","A"));

    // Common child
    DirectedMultiGraph<String, String> graph3 = new DirectedMultiGraph<String,String>() {{
      add("A", "C", "A -> C");
      add("B", "C", "B -> C");
      add("B", "D", "B -> D");
    }};
    // Assertions
    CollectionUtils.mergeVertices(graph3, "A", "B", edgeRewriter);
    Assert.assertEquals(new LinkedHashSet<String>() {{
        add("A");
        add("C");
        add("D");
      }},
      graph3.getAllVertices());
    Assert.assertEquals(new LinkedHashSet<String>() {{ add("C"); add("D"); }},
            graph3.getChildren("A") );
    Assert.assertEquals(new LinkedHashSet<String>() {{ add("A -> D"); add("A -> C"); }},
            new LinkedHashSet<>(graph3.getOutgoingEdges("A")));

    // Common parent
    DirectedMultiGraph<String, String> graph4 = new DirectedMultiGraph<String,String>() {{
      add("C", "A", "C -> A");
      add("C", "B", "C -> B");
      add("D", "B", "D -> B");
    }};
    // Assertions
    CollectionUtils.mergeVertices(graph4, "A", "B", edgeRewriter);
    Assert.assertEquals(new LinkedHashSet<String>() {{
          add("A");
          add("C");
          add("D");
        }},
        graph4.getAllVertices());
    Assert.assertEquals(new LinkedHashSet<String>() {{ add("C"); add("D"); }},
            graph4.getParents("A") );
    // The ordering is iffy really.
    Assert.assertEquals(new LinkedHashSet<String>() {{ add("D -> A"); add("C -> A"); }},
            new LinkedHashSet<>(graph4.getIncomingEdges("A")));

    // Merging the same node.
    DirectedMultiGraph<String, String> graph5 = new DirectedMultiGraph<String,String>() {{
      add("A", "B", "A -> B");
      add("C", "D", "C -> D");
    }};
    // Assertions
    CollectionUtils.mergeVertices(graph5, "A", "B", edgeRewriter);
    CollectionUtils.mergeVertices(graph5, "D", "C", edgeRewriter);
    Assert.assertEquals(new LinkedHashSet<String>() {{
        add("A");
        add("D");
      }},
      graph5.getAllVertices());
    Assert.assertEquals(0, graph5.getAllEdges().size() );

    // Gabor's cases
    DirectedMultiGraph<String, String> graph6 = new DirectedMultiGraph<String,String>() {{
      add("T", "B", "T -> B");
      add("T", "C", "T -> C");
      add("A", "T", "A -> T");
    }};
    // Assertions
    CollectionUtils.mergeVertices(graph6, "T", "A", edgeRewriter);
    Assert.assertEquals(new LinkedHashSet<String>() {{
          add("T");
          add("B");
          add("C");
        }},
        graph6.getAllVertices());
    Assert.assertEquals(2, graph6.getAllEdges().size() );
  }

  @Test
  public void testPermutations() {
    assertEquals(new LinkedHashSet<List<String>>() {{
      add(new ArrayList<String>() {{ add("a"); add("b"); }});
      add(new ArrayList<String>() {{ add("b"); add("a"); }});
    }}, CollectionUtils.permutations(new ArrayList<String>(){{ add("a"); add("b"); }}));
    assertEquals(new LinkedHashSet<List<String>>() {{
      add(new ArrayList<String>() {{ add("a"); add("b"); add("c"); }});
      add(new ArrayList<String>() {{ add("a"); add("c"); add("b"); }});
      add(new ArrayList<String>() {{ add("b"); add("a"); add("c"); }});
      add(new ArrayList<String>() {{ add("b"); add("c"); add("a"); }});
      add(new ArrayList<String>() {{ add("c"); add("a"); add("b"); }});
      add(new ArrayList<String>() {{ add("c"); add("b"); add("a"); }});
    }}, CollectionUtils.permutations(new ArrayList<String>(){{ add("c"); add("b"); add("a"); }}));
  }

  @Test
  public void testCanonicallyOrder() {
    Set<List<String>> canonicalVersions = new LinkedHashSet<>();
    for (List<String> permutation : CollectionUtils.permutations(new ArrayList<String>(){{ add("c"); add("b"); add("a"); }})) {
      canonicalVersions.add(CollectionUtils.canonicallyOrder(permutation));
    }
    assertEquals(1, canonicalVersions.size());
  }

  @Test
  public void testTransitiveClosure() {
    // A characteristic case
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(0); add(1); add(2); add(3); add(4); add(5); add(9); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{
          add(new LinkedHashSet<Integer>(){{ add(0); add(1); add(2); }});
          add(new LinkedHashSet<Integer>(){{ add(1); add(3); }});
          add(new LinkedHashSet<Integer>(){{ add(3); add(4); add(5); }});
          add(new LinkedHashSet<Integer>(){{ add(6); add(7); }});
          add(new LinkedHashSet<Integer>(){{ add(7); add(8); }});
          add(new LinkedHashSet<Integer>(){{ add(2); add(9); }});
          add(new LinkedHashSet<Integer>(){{ add(4); add(9); }});
        }}, 0));
    // Seed is not in the first set
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(0); add(1); add(2); add(3); add(4); add(5); add(9); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{
          add(new LinkedHashSet<Integer>(){{ add(0); add(1); add(2); }});
          add(new LinkedHashSet<Integer>(){{ add(1); add(3); }});
          add(new LinkedHashSet<Integer>(){{ add(3); add(4); add(5); }});
          add(new LinkedHashSet<Integer>(){{ add(6); add(7); }});
          add(new LinkedHashSet<Integer>(){{ add(7); add(8); }});
          add(new LinkedHashSet<Integer>(){{ add(2); add(9); }});
          add(new LinkedHashSet<Integer>(){{ add(4); add(9); }});
        }}, 9));
    // Some corner cases
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(0); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{ }}, 0));
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(0); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{
          add(new LinkedHashSet<Integer>(){{ add(0); add(0); }});
          add(new LinkedHashSet<Integer>(){{ add(1); add(3); }});
          add(new LinkedHashSet<Integer>(){{ add(3); add(4); add(5); }});
        }}, 0));
    // Some fuzz testing
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(6); add(7); add(8); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>() {{
          add(new LinkedHashSet<Integer>() {{
            add(0);
            add(1);
            add(2);
          }});
          add(new LinkedHashSet<Integer>() {{
            add(1);
            add(3);
          }});
          add(new LinkedHashSet<Integer>() {{
            add(3);
            add(4);
            add(5);
          }});
          add(new LinkedHashSet<Integer>() {{
            add(6);
            add(7);
          }});
          add(new LinkedHashSet<Integer>() {{
            add(7);
            add(8);
          }});
          add(new LinkedHashSet<Integer>() {{
            add(2);
            add(9);
          }});
          add(new LinkedHashSet<Integer>() {{
            add(4);
            add(9);
          }});
        }}, 6));
  }

  @Test
  public void testTransitiveClosureWithDepth() {
    assertEquals(
        new LinkedHashSet<Integer>(){{}},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{
          add(new LinkedHashSet<Integer>(){{ add(0); add(1); add(2); }});
          add(new LinkedHashSet<Integer>(){{ add(1); add(3); }});
          add(new LinkedHashSet<Integer>(){{ add(3); add(4); add(5); }});
          add(new LinkedHashSet<Integer>(){{ add(6); add(7); }});
          add(new LinkedHashSet<Integer>(){{ add(7); add(8); }});
          add(new LinkedHashSet<Integer>(){{ add(2); add(9); }});
          add(new LinkedHashSet<Integer>(){{ add(4); add(9); }});
        }}, 0, 0));
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(0); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{
          add(new LinkedHashSet<Integer>(){{ add(0); add(1); add(2); }});
          add(new LinkedHashSet<Integer>(){{ add(1); add(3); }});
          add(new LinkedHashSet<Integer>(){{ add(3); add(4); add(5); }});
          add(new LinkedHashSet<Integer>(){{ add(6); add(7); }});
          add(new LinkedHashSet<Integer>(){{ add(7); add(8); }});
          add(new LinkedHashSet<Integer>(){{ add(2); add(9); }});
          add(new LinkedHashSet<Integer>(){{ add(4); add(9); }});
        }}, 0, 1));
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(0); add(1); add(2); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{
          add(new LinkedHashSet<Integer>(){{ add(0); add(1); add(2); }});
          add(new LinkedHashSet<Integer>(){{ add(1); add(3); }});
          add(new LinkedHashSet<Integer>(){{ add(3); add(4); add(5); }});
          add(new LinkedHashSet<Integer>(){{ add(6); add(7); }});
          add(new LinkedHashSet<Integer>(){{ add(7); add(8); }});
          add(new LinkedHashSet<Integer>(){{ add(2); add(9); }});
          add(new LinkedHashSet<Integer>(){{ add(4); add(9); }});
        }}, 0, 2));
    assertEquals(
        new LinkedHashSet<Integer>(){{  add(0); add(1); add(2); add(3); add(9); }},
        CollectionUtils.transitiveClosure(new ArrayList<Set<Integer>>(){{
          add(new LinkedHashSet<Integer>(){{ add(0); add(1); add(2); }});
          add(new LinkedHashSet<Integer>(){{ add(1); add(3); }});
          add(new LinkedHashSet<Integer>(){{ add(3); add(4); add(5); }});
          add(new LinkedHashSet<Integer>(){{ add(6); add(7); }});
          add(new LinkedHashSet<Integer>(){{ add(7); add(8); }});
          add(new LinkedHashSet<Integer>(){{ add(2); add(9); }});
          add(new LinkedHashSet<Integer>(){{ add(4); add(9); }});
        }}, 0, 3));

  }

  @Test
  public void testMapIgnoreNull() {
    List<Integer> input = Arrays.asList(1, 2, 3, 4, null, 5, 6, null, null, null, 7);
    Iterator<Integer> output = CollectionUtils.mapIgnoreNull(input.iterator(), in -> in + 1);
    assertTrue(output.hasNext());
    assertEquals(2, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(3, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(4, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(5, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(6, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(7, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(8, (int) output.next());
  }

  @Test
  public void testParMapIgnoreNull() {
    List<Integer> input = Arrays.asList(1, 2, 3, 4, 5, 6, 7);
    Iterator<Integer> output = CollectionUtils.parMapIgnoreNull(input.iterator(), in -> in + 1);
    assertTrue(output.hasNext());
    assertEquals(2, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(3, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(4, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(5, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(6, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(7, (int) output.next());
    assertTrue(output.hasNext());
    assertEquals(8, (int) output.next());
    assertFalse(output.hasNext());
  }

  @Test
  public void testParMapIgnoreNullStressTest() {
    List<Integer> input = new ArrayList<>();
    int val = 0;
    for (int i = 0; i < 1000; ++i) {
      if (new Random().nextBoolean()) { input.add(val); val += 1; } else { input.add(null); }
    }
    Iterator<Integer> output = CollectionUtils.parMapIgnoreNull(input.iterator(), in -> in + 1);
    for (int i = 0; i < val; ++i) {
      assertTrue(output.hasNext());
      assertEquals(i + 1, (int) output.next());
    }
    assertFalse(output.hasNext());
  }

  @Test
  public void testParMapIgnoreNullWithWaits() {
    List<Integer> input = new ArrayList<>();
    int val = 0;
    for (int i = 0; i < 1000; ++i) {
      if (new Random().nextBoolean()) { input.add(val); val += 1; } else { input.add(null); }
    }
    Iterator<Integer> output = CollectionUtils.parMapIgnoreNull(input.iterator(), in -> {
      try {
        Thread.sleep(new Random().nextInt(5));
      } catch (InterruptedException ignored) { }
      return in + 1;
    });
    for (int i = 0; i < val; ++i) {
      assertTrue(output.hasNext());
      assertEquals(i + 1, (int) output.next());
    }
    assertFalse(output.hasNext());
  }

  @Test
  public void testTakePairs() {
    Integer[][] array = new Integer[][]{
        {1, 2, 3},
        {4, 5, 2, 3, 6},
        {7, 8, 1, 2, 9}
    };
    Function<Pair<Integer, Integer>, Boolean> condition = in -> !in.first.equals(in.second);
    Iterator<Pair<Integer, Integer>> iter = CollectionUtils.takePairs(array, condition);
    assertTrue(iter.hasNext());
    int i = 0;
    while (iter.hasNext()) {
      Pair<Integer, Integer> next = iter.next();
      assertFalse(next.first.equals(next.second));
      i += 1;
    }
    assertEquals(50, i);
  }

  @Test
  public void testBufferIteratorSimple() {
    ArrayList<Integer> elements = new ArrayList<>();
    for (int i = 0; i < 10000; ++i) {
      elements.add(i);
    }
    Iterator<Integer> iter = CollectionUtils.buffer(elements.iterator(), 100);
    for (int i = 0; i < 10000; ++i) {
      assertTrue(iter.hasNext());
      assertEquals(Integer.valueOf(i), iter.next());
    }
    assertFalse(iter.hasNext());
  }


  @Test
  public void testBufferIteratorMultithread() throws InterruptedException {
    // Create elements
    final ArrayList<Integer> elements = new ArrayList<>();
    for (int i = 0; i < 10000; ++i) {
      elements.add(i);
    }
    // Create slow iterator
    Iterator<Integer> input = new Iterator<Integer>() {
      Iterator<Integer> impl = elements.iterator();
      @Override
      public boolean hasNext() { return impl.hasNext(); }
      @Override
      public Integer next() {
        Thread.yield();
        return impl.next();
      }
      @Override
      public void remove() { }
    };
    // Buffer
    final Iterator<Integer> iter = CollectionUtils.buffer(input, 100);
    // Read multithreaded
    List<Thread> threads = new ArrayList<>();
    for (int i = 0; i < 100; ++i) {
      Thread t = new Thread() {
        @Override
        public void run() {
          int last = -1;
          for (int i = 0; i < 100; ++i) {
            assertTrue(iter.hasNext());
            int next = iter.next();
            assertTrue( next > last );
            last = next;
          }
        }
      };
      t.start();
      threads.add(t);
    }
    // Join threads
    for (Thread t : threads) {
      t.join();
    }
    assertFalse(iter.hasNext());
  }

  @Test
  public void testTake() {
    // Create elements
    final ArrayList<Integer> elements = new ArrayList<>();
    for (int i = 0; i < 7; ++i) {
      elements.add(i);
    }

    // Trivial case
    IterableIterator<Integer> take5 = CollectionUtils.take(elements, 5);
    assertTrue(take5.hasNext()); assertEquals(new Integer(0), take5.next());
    assertTrue(take5.hasNext()); assertEquals(new Integer(1), take5.next());
    assertTrue(take5.hasNext()); assertEquals(new Integer(2), take5.next());
    assertTrue(take5.hasNext()); assertEquals(new Integer(3), take5.next());
    assertTrue(take5.hasNext()); assertEquals(new Integer(4), take5.next());
    assertFalse(take5.hasNext());

    // Corner case: empty
    IterableIterator<Integer> take0 = CollectionUtils.take(elements, 0);
    assertFalse(take0.hasNext());

    // Corner case: overflow
    IterableIterator<Integer> take100 = CollectionUtils.take(elements, 100);
    for (int i = 0; i < 7; ++i) {
      assertTrue(take100.hasNext());
      take100.next();
    }
    assertFalse(take100.hasNext());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testConcat() {
    Iterable<Integer> collection1 = new ArrayList<Integer>() {{ add(1); add(2); add(3); }};
    Iterable<Integer> collection2 = new ArrayList<Integer>() {{ add(4); add(5); add(6); }};
    Iterable<Integer> collection3 = new ArrayList<Integer>() {{ add(7); }};
    Iterable<Integer> collection4 = new ArrayList<Integer>() {{  }};
    Iterator<Integer> output = CollectionUtils.concat(collection1, collection2, collection3, collection4).iterator();

    assertTrue(output.hasNext()); assertEquals(new Integer(1), output.next());
    assertTrue(output.hasNext()); assertEquals(new Integer(2), output.next());
    assertTrue(output.hasNext()); assertEquals(new Integer(3), output.next());
    assertTrue(output.hasNext()); assertEquals(new Integer(4), output.next());
    assertTrue(output.hasNext()); assertEquals(new Integer(5), output.next());
    assertTrue(output.hasNext()); assertEquals(new Integer(6), output.next());
    assertTrue(output.hasNext()); assertEquals(new Integer(7), output.next());
    assertFalse(output.hasNext());
  }

}
