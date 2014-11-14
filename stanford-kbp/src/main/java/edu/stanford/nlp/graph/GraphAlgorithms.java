package edu.stanford.nlp.graph;

import edu.stanford.nlp.util.BinaryHeapPriorityQueue;

import java.util.*;

/**
 * A collection of useful graph algorithms
 */
public class GraphAlgorithms {

  // TODO(Arun): I have a bug in me.
  public static<V, E> Map<V, Double> getDistances(Graph<V,E> graph, V root, int threshold) {
    BinaryHeapPriorityQueue<V> open = new BinaryHeapPriorityQueue<V>();
    open.add(root, 0);
    Map<V,Double> closed = new HashMap<V,Double>();

    while(!open.isEmpty()) {
      double distance = open.getPriority();
      V top = open.removeFirst();
      assert !closed.containsKey(top);

      closed.put(top, -distance);
      // Undirected
      if(-distance < threshold) {
        for(V neighbor : graph.getNeighbors(top)) {
          if(!closed.containsKey(neighbor))
            if(open.contains(neighbor)) open.changePriority(neighbor, Math.min(open.getPriority(neighbor),distance+1));
            else open.add(neighbor, distance-1); // The priorty heap chooses the largest value.
        }
      }
    }

    return closed;
  }

  public static <V,E> Map<V,Map<V,Double>> getMutualDistances(Graph<V,E> graph, int threshold) {
    // Initialize
    Map<V,Map<V,Double>> adj = new HashMap<V,Map<V,Double>>();
    for(V vertex : graph.getAllVertices())
      adj.put(vertex, getDistances(graph, vertex, threshold));
    return adj;
  }

}
