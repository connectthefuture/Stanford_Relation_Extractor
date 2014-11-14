package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.graph.DirectedMultiGraph;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;

import java.lang.reflect.Array;
import java.util.*;
import java.util.PriorityQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Utilities for common collection types
 */
@SuppressWarnings("UnusedDeclaration")
public class CollectionUtils {

  public static <T1,T2> Pair<List<T1>,List<T2>> unzip(List<Pair<T1,T2>> lst) {
    int elems = lst.size();
    List<T1> lst1 = new ArrayList<>(elems);
    List<T2> lst2 = new ArrayList<>(elems);

    for( Pair<T1,T2> element : lst ) {
      lst1.add(element.first);
      lst2.add(element.second);
    }
    return new Pair<>(lst1, lst2);
  }
  public static <T1,T2> List<Pair<T1,T2>> zip(List<T1> lst1, List<T2> lst2 ) {
    int elems = (lst1.size() < lst2.size()) ? lst1.size() : lst2.size();
    List<Pair<T1,T2>> lst = new ArrayList<>(elems);

    for(int i = 0; i < elems; i++) {
      lst.add( new Pair<>( lst1.get(i), lst2.get(i) ) );
    }
    return lst;
  }



  public static <T1> Maybe<T1> find( Collection<T1> lst, Function<T1,Boolean> filter) {
    for( T1 elem : lst )
      if(filter.apply(elem)) return Maybe.Just(elem);
    return Maybe.Nothing();
  }
  public static <T1> boolean exists( Collection<T1> lst, Function<T1,Boolean> filter) {
    for( T1 elem : lst )
      if(filter.apply(elem)) return true;
    return false;
  }
  public static <T1> boolean forall( Collection<T1> lst, Function<T1,Boolean> filter) {
    for( T1 elem : lst )
      if(!filter.apply(elem)) return false;
    return true;
  }

  public static <T1,T2> List<T2> map( Iterable<T1> lst, Function<T1,T2> mapper) {
    List<T2> lst_ = new ArrayList<>();
    for( T1 elem : lst )
      lst_.add( mapper.apply(elem) );
    return lst_;
  }

  public static <T1,T2> List<T2> filterMap( Iterable<T1> lst, Function<T1,Maybe<T2>> mapper) {
    List<T2> lst_ = new ArrayList<>();
    for( T1 elem : lst )
      for( T2 elem_ : mapper.apply(elem) )
        lst_.add( elem_ );
    return lst_;
  }

  @SuppressWarnings("unchecked")
  public static <T1,T2> T2[] map( T1[] lst, Function<T1,T2> mapper) {
    List<T2> lst_ = new ArrayList<>(lst.length);
    for( T1 elem : lst ) {
      lst_.add( mapper.apply(elem) );
    }
    if (lst.length == 0) { return (T2[]) new Object[0]; }
    T2[] rtn = (T2[]) Array.newInstance(lst_.get(0).getClass(), lst_.size());
    return lst_.toArray(rtn);
  }

  public static <T1,T2> List<T2> lazyMap( final List<T1> lst, final Function<T1,T2> mapper) {
    if (lst == null) { return null; }
    return new AbstractList<T2>() {
      @Override
      public T2 get(int index) {
        return mapper.apply(lst.get(index));
      }
      @Override
      public int size() {
        return lst.size();
      }
    };
  }

  public static <T1,T2> Collection<T2> lazyMap( final Collection<T1> lst, final Function<T1,T2> mapper) {
    return new AbstractCollection<T2>() {
      @Override
      public Iterator<T2> iterator() {
        return new Iterator<T2>() {
          Iterator<T1> impl = lst.iterator();
          @Override
          public boolean hasNext() { return impl.hasNext(); }
          @Override
          public T2 next() { return mapper.apply(impl.next()); }
          @Override
          public void remove() { impl.remove(); }
        };
      }
      @Override
      public int size() {
        return lst.size();
      }
    };
  }

  public static <T1,T2> IterableIterator<T2> mapIgnoreNull( final Iterator<T1> lst, final Function<T1,T2> mapper) {
    return new IterableIterator<>(new Iterator<T2>(){
      private T2 next = null;
      @Override
      public boolean hasNext() {
        if (next == null) {
          while (lst.hasNext() && next == null) {
            T1 nextIn = lst.next();
            if (nextIn != null) { next = mapper.apply(nextIn); }
          }
          return next != null;
        } else {
          return true;
        }
      }
      @Override
      public T2 next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        T2 rtn = next;
        assert rtn != null;
        next = null;
        return rtn;
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }

  /**
   * Map from one iterator to another in parallel, ignoring null entries.
   * The output is guaranteed to be in the same order as the input.
   * Note, however, that the overhead for starting every task is relatively large for this
   * method, and thus it should be used primarily for long methods.
   *
   * @see edu.stanford.nlp.kbp.common.CollectionUtils#mapIgnoreNull(java.util.Iterator, java.util.function.Function)
   */
  public static <T1,T2> IterableIterator<T2> parMapIgnoreNull(  Iterator<T1> lst, final Function<T1,T2> mapper) {
    final BlockingQueue<Maybe<T1>> workQueue = new ArrayBlockingQueue<>(3 * Execution.threads);
    final BlockingQueue<Maybe<T2>> resultQueue = new ArrayBlockingQueue<>(3 * Execution.threads);
    final Iterator<T1> iter = mapIgnoreNull(lst, in -> in);
    // Worker thread
    Thread worker = new Thread() {
      ExecutorService exec = Executors.newFixedThreadPool(Execution.threads);
      @Override
      public void run() {
        // Run jobs
        try {
          Maybe<T1> task;
          SimpleLock mutableLastTaskDone = new SimpleLock();
          while ( (task = workQueue.take()).isDefined() ) {
            // Ensure synchronous writes
            final SimpleLock lastTaskDone = mutableLastTaskDone;
            final SimpleLock thisTaskDone = new SimpleLock();
            thisTaskDone.acquire();
            mutableLastTaskDone = thisTaskDone;
            // Get the task
            final T1 input = task.get();
            // Run the mapper
            exec.submit(() -> {
              try {
                // Run the computation
                final T2 result = mapper.apply(input);
                // Free this thread, and add the elements in a new thread
                Thread adder = new Thread() {
                  @Override
                  public void run() {
                    lastTaskDone.acquire();  // make sure the last task has finished
                    try {
                      resultQueue.put(Maybe.Just(result));
                    } catch (InterruptedException e) {
                      e.printStackTrace();
                    } finally {
                      lastTaskDone.release();  // (release last task, just so everything ends unlocked)
                      thisTaskDone.release();  // release this task
                    }
                  }
                };
                adder.setDaemon(true);
                adder.start();
              } catch (Throwable t) {
                t.printStackTrace();
              }
            });
          }
          // Signal end-of-jobs
          mutableLastTaskDone.acquire();
          mutableLastTaskDone.release();
          resultQueue.put(Maybe.<T2>Nothing());
        } catch (InterruptedException ignored) { }
      }
    };
    worker.setDaemon(true);  // Kill on program termination
    worker.start();

    // The iterator interface returned
    return new IterableIterator<>(new Iterator<T2>(){
      // Nothing means no cached element; null means no more elements ever
      private Maybe<T2> next = Maybe.Nothing();

      @Override
      public boolean hasNext() {
        if (next == null) { return false; }
        if (next.isDefined()) { return true; }

        // Fill up the work queue
        while (workQueue.remainingCapacity() > 0) {
          if (iter.hasNext()) {
            workQueue.add(Maybe.Just(iter.next()));
          } else {
            workQueue.add(Maybe.<T1>Nothing());
            break;
          }
        }

        // Poll from the queue
        try {
          Maybe<T2> result = resultQueue.take();
          if (result.isDefined()) {
            next = result;
          } else {
            next = null;
          }
        } catch (InterruptedException ignored) { }
        return next != null;
      }
      @Override
      public T2 next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        T2 rtn = next.orCrash();
        assert rtn != null;
        next = Maybe.Nothing();
        return rtn;
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }

  /** @see CollectionUtils#parMapIgnoreNull(java.util.Iterator, java.util.function.Function) */
  public static <T1,T2> IterableIterator<T2> parMapIgnoreNullUnordered(  Iterator<T1> lst, final Function<T1,T2> mapper) {
    final BlockingQueue<Maybe<T1>> workQueue = new ArrayBlockingQueue<>(3 * Execution.threads);
    final BlockingQueue<Maybe<T2>> resultQueue = new ArrayBlockingQueue<>(3 * Execution.threads);
    final Iterator<T1> iter = mapIgnoreNull(lst, in -> in);
    // Worker thread
    Thread worker = new Thread() {
      ExecutorService exec = Executors.newFixedThreadPool(Execution.threads);
      @Override
      public void run() {
        // Run jobs
        try {
          Maybe<T1> task;
          while ( (task = workQueue.take()).isDefined() ) {
            // Get the task
            final T1 input = task.get();
            // Run the mapper
            exec.submit(() -> {
              try {
                // Run the computation
                final T2 result = mapper.apply(input);
                resultQueue.put(Maybe.Just(result));
              } catch (Throwable t) {
                t.printStackTrace();
              }
            });
          }
          // Signal end-of-jobs
          resultQueue.put(Maybe.<T2>Nothing());
        } catch (InterruptedException ignored) { }
      }
    };
    worker.setDaemon(true);  // Kill on program termination
    worker.start();

    // The iterator interface returned
    for(int i=0; i< Execution.threads*3; i++) {
      if (iter.hasNext()) {
        workQueue.add(Maybe.Just(iter.next()));
      } else {
        workQueue.add(Maybe.<T1>Nothing());
        break;
      }
    }
    return new IterableIterator<>(new Iterator<T2>(){
      // Nothing means no cached element; null means no more elements ever
      private Maybe<T2> next = Maybe.Nothing();

      @Override
      public boolean hasNext() {
        if (next == null) { return false; }
        if (next.isDefined()) { return true; }

        // Fill up the work queue

        if (iter.hasNext()) {
          workQueue.add(Maybe.Just(iter.next()));
        } else {
          workQueue.add(Maybe.<T1>Nothing());
        }


        // Poll from the queue
        try {
          Maybe<T2> result = resultQueue.take();
          if (result.isDefined()) {
            next = result;
          } else {
            next = null;
          }
        } catch (InterruptedException ignored) { }
        return next != null;
      }
      @Override
      public T2 next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        T2 rtn = next.orCrash();
        assert rtn != null;
        next = Maybe.Nothing();
        return rtn;
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }

  public static <T1,T2> IterableIterator<T2> flatMapIgnoreNull( final Iterator<T1> input, final Function<T1,Iterator<T2>> mapper) {
    return new IterableIterator<>(new Iterator<T2>(){
      private Iterator<T2> iter = null;
      private T2 next = null;

      private boolean hasNextInIter() {
        if (iter == null) { return false; }
        if (next == null) {
          while (iter.hasNext() && next == null) {
            next = iter.next();
          }
          return next != null;
        } else {
          return true;
        }
      }

      @Override
      public boolean hasNext() {
        while (!hasNextInIter()) {
          if (!input.hasNext()) { return false; }
          iter = mapper.apply(input.next());
        }
        return iter != null;
      }
      @Override
      public T2 next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        T2 toReturn = next;
        next = null;
        return toReturn;
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }


  @SuppressWarnings("unchecked")
  public static <T> Iterable<T> concat(final Iterable<T>... collections) {
    return new Iterable<T> () {
      @Override
      public Iterator<T> iterator() {
        return new Iterator<T>() {
          final Iterable<T>[] iterables = collections;
          int iteratorIdx = 0;
          Iterator<T> it = collections[0].iterator();

          private synchronized void advanceCursor() {
            while(!it.hasNext() && iteratorIdx < iterables.length) {
              iteratorIdx += 1;
              if (iteratorIdx == iterables.length) {
                it = new Iterator<T>() {
                  @Override
                  public boolean hasNext() { return false; }
                  @Override
                  public T next() { throw new NoSuchElementException(); }
                  @Override
                  public void remove() { throw new RuntimeException(); }
                };
              } else {
                it = iterables[iteratorIdx].iterator();
              }
            }
          }

          @Override
          public synchronized boolean hasNext() {
            advanceCursor();
            return iteratorIdx != iterables.length && it.hasNext();
          }

          @Override
          public synchronized T next() {
            advanceCursor();
            if(iteratorIdx == iterables.length)
              throw new NoSuchElementException();
            return it.next();
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
      }
    };
  }


  public static <T1,T2> List<T2> concat( List<T1> lst, Function<T1,List<T2>> mapper) {
    List<T2> lst_ = new ArrayList<>(lst.size());
    for( T1 elem : lst )
      lst_.addAll(mapper.apply(elem));
    return lst_;
  }
  public static <T1,T2,T3> Map<T2,List<T3>> collect( List<T1> lst, Function<T1,Pair<T2,T3>> mapper) {
    Map<T2,List<T3>> map = new HashMap<>();
    for( T1 elem : lst ) {
      Pair<T2,T3> pair = mapper.apply(elem);
      if( !map.containsKey(pair.first) )
        map.put(pair.first, new ArrayList<T3>() );
      map.get(pair.first).add(pair.second);
    }
    return map;
  }
  public static <T1,T2,T3> Map<T2,Set<T3>> collectDistinct( List<T1> lst, Function<T1,Pair<T2,T3>> mapper) {
    Map<T2,Set<T3>> map = new HashMap<>();
    for( T1 elem : lst ) {
      Pair<T2,T3> pair = mapper.apply(elem);
      if( !map.containsKey(pair.first) )
        map.put(pair.first, new HashSet<T3>() );
      map.get(pair.first).add(pair.second);
    }
    return map;
  }

  public static <T1> IterableIterator<T1> filter(final Iterator<T1> iter, final Function<T1,Boolean> filter) {
    return new IterableIterator<>(new Iterator<T1>() {
      private T1 next = null;
      @Override
      public boolean hasNext() {
        while (next == null && iter.hasNext()) {
          next = iter.next();
          if (!filter.apply(next)) { next = null; }
        }
        return next != null;
      }
      @Override
      public T1 next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        T1 rtn = next;
        next = null;
        return rtn;
      }
      @Override
      public void remove() {
        iter.remove();
      }
    });
  }

  public static <T1> List<T1> filter(Collection<T1> lst, Function<T1,Boolean> filter) {
    List<T1> output = new ArrayList<>();
    for(T1 elem : lst) {
      if (filter.apply(elem)) output.add(elem);
    }
    return output;
  }

  public static <T1> Maybe<T1> find( Collection<T1> lst, T1 thing, Function<Pair<T1,T1>,Boolean> comparator ) {
    for( T1 elem : lst ) {
      if( comparator.apply( Pair.makePair(elem, thing) ) ) return Maybe.Just( thing );
    }
    return Maybe.Nothing();
  }

  public static <T1> Maybe<T1> overlap( Collection<T1> lst, Collection<T1> lst_ ) {
    for( T1 elem : lst ) {
      if( lst_.contains( elem ) ) return Maybe.Just(elem);
    }
    return Maybe.Nothing();
  }

  public static <T1> List<T1> allOverlaps( Collection<T1> lst, Collection<T1> lst_ ) {
    List<T1> overlaps = new ArrayList<>();
    for( T1 elem : lst ) {
      if( lst_.contains( elem ) ) overlaps.add( elem );
    }
    return overlaps;
  }

  /**
   * Partition on the keys returned by mapper
   */
  public static <T1,T2> Map<T2,List<T1>> partition( List<T1> lst, Function<T1,T2> mapper) {
    Map<T2,List<T1>> map = new HashMap<>();
    for( T1 elem : lst ) {
      T2 key = mapper.apply(elem);
      if( !map.containsKey(key) )
        map.put(key, new ArrayList<T1>() );
      map.get(key).add(elem);
    }
    return map;
  }

  // Removes all elements of lst2 from lst1
  public static <T1> void difference( List<T1> lst1, List<T1> lst2 ) {
    for(T1 elem : lst2) {
        lst1.remove(elem);
    }
  }

  /**
   * Create a sub list with just these indices
   */
  public static <T1> List<T1> subList( List<T1> lst, Collection<Integer> indices ) {
    List<T1> sublst = new ArrayList<>();
    for(Integer idx : indices)
      sublst.add(lst.get(idx));
    return sublst;
  }

  // -- Graph functions

  /**
   * Map each edge of the graph, retaining vertices
   */
  public static <T1,T2,T3> DirectedMultiGraph<T1,T3> mapEdges( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,T2>,T3> mapper ) {
      DirectedMultiGraph<T1,T3>  graph_ = new DirectedMultiGraph<>();
      for( T1 head : graph.getAllVertices() ) {
          for( T1 tail : graph.getChildren(head) ) {
              for( T2 edge : graph.getEdges(head, tail) ) {
                  graph_.add(head, tail, mapper.apply(Triple.makeTriple(head, tail, edge)));
              }
          }
      }
      return graph_;
  }

  /**
   * Map the set of edges between vertices of a graph, retaining vertices
   */
  public static <T1,T2,T3> DirectedMultiGraph<T1,T3> mapEdgeSets( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,List<T2>>,List<T3>> mapper ) {
    DirectedMultiGraph<T1,T3>  graph_ = new DirectedMultiGraph<>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        for( T3 edge_ : mapper.apply(Triple.makeTriple(head, tail, graph.getEdges(head, tail)))) {
          graph_.add(head, tail, edge_);
        }
      }
    }
    return graph_;
  }

  public static <T1,T2,T3,T4> DirectedMultiGraph<T3,T4> map( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,List<T2>>,Triple<T3,T3,List<T4>> > mapper ) {
    DirectedMultiGraph<T3,T4>  graph_ = new DirectedMultiGraph<>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        Triple<T3,T3,List<T4>> triple = mapper.apply(Triple.makeTriple(head, tail, graph.getEdges(head, tail)));
        for( T4 edge : triple.third ) {
          graph_.add(triple.first, triple.second, edge);
        }
      }
    }
    return graph_;
  }

  /**
   * Similar to map, but allows you to return a list, adding many edges for each edge between two vertices
   */
  public static <T1,T2,T3> DirectedMultiGraph<T1,T3> collectEdges( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,T2>,List<T3>> mapper ) {
    DirectedMultiGraph<T1,T3>  graph_ = new DirectedMultiGraph<>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        for( T2 edge : graph.getEdges(head, tail) ) {
          for( T3 edge_ : mapper.apply(Triple.makeTriple(head, tail, edge))) {
            graph_.add(head, tail, edge_);
          }
        }
      }
    }
    return graph_;
  }


  public static <T1,T2> List<Pair<T1,T1>> vertexPairs( DirectedMultiGraph<T1,T2> graph ) {
    List<Pair<T1,T1>> pairs = new ArrayList<>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        pairs.add( Pair.makePair(head, tail) );
      }
    }
    return pairs;
  }

  public static <T1,T2> List<Triple<T1,T1,List<T2>>> groupedEdges( DirectedMultiGraph<T1,T2> graph ) {
    List<Triple<T1,T1,List<T2>>> edges = new ArrayList<>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        edges.add( Triple.makeTriple(head, tail, graph.getEdges(head, tail)) );
      }
    }
    return edges;
  }

  public static <T> boolean equalOrBothUndefined(Maybe<T> x, Maybe<T> y) {
    if (x.isDefined() && y.isDefined()) {
      if (x.get() == null && y.get() == null) { assert false; return true; }
      else if (x.get() == null || y.get() == null) { assert false; return false; }
      else {
        if (Double.class.isAssignableFrom(x.get().getClass()) &&
            Double.class.isAssignableFrom(y.get().getClass())) {
          return Math.abs(((Double) x.get()) - ((Double) y.get())) < 1e-5;
        } else {
          return x.get().equals(y.get());
        }
      }
    } else return !x.isDefined() && !y.isDefined();
  }

  public static <T> boolean equalIfBothDefined(Maybe<T> x, Maybe<T> y) {
    if (x.isDefined() && y.isDefined()) {
      if (x.get() == null && y.get() == null) { assert false; return true; }
      else if (x.get() == null || y.get() == null) { assert false; return false; }
      else {
        if (Double.class.isAssignableFrom(x.get().getClass()) &&
            Double.class.isAssignableFrom(y.get().getClass())) {
          return Math.abs(((Double) x.get()) - ((Double) y.get())) < 1e-5;
        } else {
          return x.get().equals(y.get());
        }
      }
    } else {
      return true;
    }
  }

  private static <E> PriorityQueue<Triple<E, Double, Integer>> createFrontier(int size) {
    return new PriorityQueue<>(size,
        (o1, o2) -> {
          if (o1.second > o2.second) return -1;
          if (o1.second < o2.second) return 1;
          else return 0;
        }
    );
  }

  /**
   * Interleave the results from $n$ different iterators. In particular, it will return a single iterator with
   * elements going from highest to lowest priority, according to the Double value associated with each element
   * returned from the constituent iterators.
   *
   * @param elements An array of iterators to interleave
   * @param <E> The type of element returned from the iterator
   * @return A single iterator, containing the elements from the constituent iterators interleaved so that the
   *         highest priority element (of any constituent) is first, followed by the second highest, and so forth.
   */
  public static <E> IterableIterator<Pair<E, Double>> interleave(final Iterator<Pair<E, Double>>[] elements) {
    // -- Handle state
    final Set<E> queued = new HashSet<>();
    final PriorityQueue<Triple<E, Double, Integer>> frontier = createFrontier(elements.length);

    // -- Set Up Search
    PriorityQueue<Triple<E, Double, Integer>> immediateFrontier = createFrontier(elements.length);
    // Load first results
    for (int queryIndex = 0; queryIndex < elements.length; ++queryIndex) {
      if (elements[queryIndex].hasNext()) {
        Pair<E, Double> next = elements[queryIndex].next();
        immediateFrontier.add(Triple.makeTriple(next.first, next.second, queryIndex));
      }
    }
    // Queue first results
    for (Triple<E, Double, Integer> term : immediateFrontier) {
      if (!queued.contains(term.first)) {
        frontier.add(term);
        queued.add(term.first);
      }
    }

    // -- Run [lazy] Search
    return new IterableIterator<>(new Iterator<Pair<E, Double>>() {
      @Override
      public boolean hasNext() {
        return !frontier.isEmpty();
      }
      @Override
      public Pair<E, Double> next() {
        // Get next element
        Triple<E, Double, Integer> next = frontier.poll();
        int queryIndex = next.third;
        // Update queue
        boolean haveQueued = false;
        while (!haveQueued && elements[queryIndex].hasNext()) {
          Pair<E, Double> toQueue = elements[queryIndex].next();
          if (!queued.contains(toQueue.first)) {
            frontier.add(Triple.makeTriple(toQueue.first, toQueue.second, queryIndex));
            queued.add(toQueue.first);
            haveQueued = true;
          }
        }
        // Return
        return Pair.makePair(next.first, next.second);
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }

  /**
   * Interleave a number of iterators, weighting each iterator by the given quantity.
   * @param elements The elements to interleave.
   * @param relativeRatios The relative ratios to interleave the elements according to.
   * @param <E> The type of element to iterate over.
   * @return An iterator interleaving the specified iterators weighted by the given ratios.
   */
  public static <E> IterableIterator<E> interleave(
      final Iterator<E>[] elements,
      final double[] relativeRatios) {
    @SuppressWarnings("unchecked")
    Iterator<Pair<E, Double>>[] weightedElems = new Iterator[elements.length];
    for (int i = 0; i < elements.length; ++i) {
      final int index = i;
      weightedElems[i] = CollectionUtils.mapIgnoreNull(elements[i], new Function<E, Pair<E, Double>>() {
        private double weight = Double.MAX_VALUE;
        @Override
        public Pair<E, Double> apply(E in) {
          Pair <E, Double> value = Pair.makePair(in, weight);
          weight -= 1.0 / relativeRatios[index];
          return value;
        }
      });
    }
    return CollectionUtils.mapIgnoreNull(interleave(weightedElems), in -> in.first);
  }

  /**
   * Take the first N elements of a collection.
   * @param elements The collection to take the first elements from.
   * @param count The number of elements to take, from the beginning of the collection.
   * @param <E> The type of the collection.
   * @return An iterator corresponding to the first elements taken from this collection.
   */
  public static <E> IterableIterator<E> take(final Iterable<E> elements, final long count) {
    return new IterableIterator<>(new Iterator<E>(){
      final Iterator<E> iter = elements.iterator();
      int returnedCount = 0;
      E nextElement = null;

      private synchronized boolean ensureNext() {
        if (nextElement == null) {
          if (returnedCount >= count) { return false; }
          if (!iter.hasNext()) { return false; }
          nextElement = iter.next();
          returnedCount += 1;
        }
        return true;
      }

      @Override
      public boolean hasNext() {
        return ensureNext();
      }

      @Override
      public synchronized E next() {
        if (!ensureNext()) { throw new NoSuchElementException(); }
        E next = nextElement;
        nextElement = null;
        return next;
      }

      @Override
      public void remove() {
        iter.remove();
      }
    });
  }

  public static int[] seq(int n) {
    int[] rtn = new int[n];
    for (int i = 0; i < n; ++i) { rtn[i] = i; }
    return rtn;
  }

  public static IterableIterator<Integer> seqIter(final int n) {
    return new IterableIterator<Integer>(new AbstractIterator<Integer>() {
      int next = 0;
      @Override
      public boolean hasNext() {
        return next < n;
      }
      @Override
      public Integer next() {
        next += 1;
        return next - 1;
      }
    });
  }

  public static <E> void shuffleInPlace(E[] elems, Random rand) {
    for(int j = elems.length - 1; j > 0; j --){
      int randIndex = rand.nextInt(j+1);
      E tmp = elems[randIndex];
      elems[randIndex] = elems[j];
      elems[j] = tmp;
    }
  }

  /**
   * Creates an iterator from a function which continually produces Maybe's.
   * Perhaps somewhat counterintuitively (but fitting the use case for its creation),
   * the semantics of an element coming out of the factory are:
   *
   * -&gt; Just(E): iterator returns E
   * -&gt; Nothing: iterator skips over this element. ***it doesn't stop on Nothing!***
   * -&gt; null:    iterator stops
   *
   * @param factory The function which creates Maybe's
   * @param <E> The return type
   * @return An iterator over non-Nothing Maybe's returned by the factory
   */
  @SuppressWarnings("UnusedDeclaration")
  public static <E> Iterator<E> iteratorFromMaybeFactory(final Factory<Maybe<E>> factory) {
    return new Iterator<E>() {

      private Maybe<E> nextElement = Maybe.Nothing();

      @Override
      public boolean hasNext() {
        if (nextElement == null) { return false; }
        if (nextElement.isDefined()) { return true; }
        while (nextElement != null && !nextElement.isDefined()) {
          nextElement = factory.create();
        }
        return nextElement != null;
      }

      @Override
      public E next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        E element = nextElement.get();
        nextElement = Maybe.Nothing();
        return element;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }

  public static <E> Iterator<E> iteratorFromMaybeIterableFactory(final Factory<Maybe<Iterable<E>>> factory) {
    return new Iterator<E>() {

      private Maybe<Iterator<E>> nextIterator = Maybe.Nothing();
      private E nextElement;

      @Override
      public boolean hasNext() {
        // Already found the next element
        if (nextElement != null) { return true; }
        // Can get the next element from the iterator
        if (nextIterator.isDefined() && nextIterator.get().hasNext()) {
          nextElement = nextIterator.get().next();
          return true;
        }
        // Update iterator state
        if (nextIterator.isDefined() && !nextIterator.get().hasNext()) {
          nextIterator = Maybe.Nothing();
        }
        // Get a new iterator
        while (nextIterator != null && !nextIterator.isDefined()) {
          Maybe<Iterable<E>> next = factory.create();
          nextIterator = next == null ? null : next.isDefined() ? Maybe.Just(next.get().iterator()) : Maybe.<Iterator<E>>Nothing();
        }
        // End of the line
        if (nextIterator == null) { return false; }
        // Else try again with a new iterator
        return hasNext();  // stack depth should be the number of Nothing iterators returned.
      }

      @Override
      public E next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        E element = nextElement;
        nextElement = null;
        return element;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }

  public static double sum(Collection<Double> collection) {
    double total = 0.;
    for(double value : collection)
      total += value;
    return total;
  }

  public static <E> Set<E> intersect(Set<E> a, Set<E> b) {
    Set<E> intersect = new HashSet<>();
    for (E entryA : a) {
      if (b.contains(entryA)) { intersect.add(entryA); }
    }
    for (E entryB : b) {
      if (a.contains(entryB)) { intersect.add(entryB); }
    }
    return intersect;
  }

  public interface EdgeRewriter<V,E> {
    public boolean sameEdge(E victorEdge, E candidateEdge);
    public boolean isValidOutgoingEdge(V victor, E edge);
    public E mergeEdges(E edge1, E edge2);
    public E rewrite(V pivot, V newValue, E edge);
  }

  /**
   * Merges the loser into the victor in the graph
   */
  public static<V,E> void mergeVertices( DirectedMultiGraph<V,E> graph, V victor, V loser,
                                         final EdgeRewriter<V,E> edgeRewriter) {
    if( victor.equals( loser ) ) return;

    startTrack("Merging vertices: " + victor + " <- " + loser );

    int originalVictorSize = graph.getChildren(victor).size();
    int originalLoserSize = graph.getChildren(loser).size();

    //V updatedVictor = edgeRewriter.mergeVertices(victor, loser);

    // 1. Copy over all edges.
    for( V child : new ArrayList<>(graph.getChildren(victor)) ) {
      if( child.equals(loser) ) continue; // Don't add self-loops
      List<E> victorEdges = new ArrayList<>(graph.getEdges( victor, child ));
      graph.removeEdges(victor, child);

      List<E> loserEdges = new ArrayList<>(graph.getEdges( loser, child ));
      graph.removeEdges(loser, child);
      for( final E victorEdge : victorEdges ) {
        Maybe<E> loserEdge = CollectionUtils.find(loserEdges, in -> edgeRewriter.sameEdge(victorEdge, in));
        if(loserEdge.isDefined()) {
          loserEdges.remove(loserEdge.get());
          graph.add( victor, child, edgeRewriter.mergeEdges(victorEdge, loserEdge.get()) );
          // Merge
        } else {
          // Add to graph
          graph.add( victor, child, victorEdge );
        }
      }
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(child, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(victor, outgoingEdge)) {
          graph.add(victor, child, outgoingEdge);
        }
      }
    }
    // Move the remaining edges of the loser
    for( V child : new ArrayList<>(graph.getChildren( loser ) ) ) {
      if( child.equals(victor) ) continue;
      List<E> loserEdges = new ArrayList<>(graph.getEdges( loser, child ));
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(child, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(victor, outgoingEdge)) {
          graph.add(victor, child, outgoingEdge);
        }
      }
    }

    // Repeat except for parent nodes
    for( V parent : new ArrayList<>(graph.getParents(victor)) ) {
      if( parent.equals(loser) ) continue; // Don't add self-loops
      List<E> victorEdges = new ArrayList<>(graph.getEdges( parent, victor ));
      graph.removeEdges(parent, victor);

      List<E> loserEdges = new ArrayList<>(graph.getEdges( parent, loser));
      graph.removeEdges(parent, loser);
      for( final E victorEdge : victorEdges ) {
        Maybe<E> loserEdge = CollectionUtils.find(loserEdges, in -> edgeRewriter.sameEdge(victorEdge, in));
        if(loserEdge.isDefined()) {
          loserEdges.remove(loserEdge.get());
          graph.add( parent, victor, edgeRewriter.mergeEdges(victorEdge, loserEdge.get()) );
          // Merge
        } else {
          // Add to graph
          graph.add( parent, victor, victorEdge );
        }
      }
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(parent, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(parent, outgoingEdge)) {
          graph.add(parent, victor, outgoingEdge);
        }
      }
    }
    // 
    for( V parent : new ArrayList<>(graph.getParents( loser ) ) ) {
      if( parent.equals(victor) ) continue;
      List<E> loserEdges = new ArrayList<>(graph.getEdges( parent, loser ));
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(parent, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(parent, outgoingEdge)) {
          graph.add(parent, victor, outgoingEdge);
        }
      }
    }

    // Now, finally delete the old vertex
    graph.removeVertex( loser );

//    assert graph.getChildren(victor).size() >= Math.min(originalVictorSize, originalLoserSize);  // note(gabor): not true, if edges are filtered.
    assert graph.getChildren(victor).size() <= originalVictorSize + originalLoserSize;

    endTrack("Merging vertices: " + victor + " <- " + loser );
  }

  /**
   * Group elements in a list together by equivalence classes
   */
  public static<V> List<Set<V>> groupByEquivalence( Collection<V> lst,
      Function<Pair<V,V>,Boolean> comparator ) {

    // Create a map of items and things they are equivalent to
    Map<V, List<V>> equivalenceMatching = new HashMap<>();
    startTrack("Making equivalence lists");
    for( V elem : lst ) {
      equivalenceMatching.put( elem, new ArrayList<V>() );
      for( V elem_ : lst ) {
        if( elem.equals( elem_ ) ) continue; // don't add yourself
        if( comparator.apply( Pair.makePair( elem, elem_ ) ) ) {
          equivalenceMatching.get( elem ).add( elem_ );
        }
      }
    }
    endTrack("Making equivalence lists");
    return groupByEquivalence(lst, equivalenceMatching);
  }

  public static <V,T> Map<T,List<V>> groupBy(List<V> lst, Function<V,T> selector) {
    Map<T,List<V>> groupedList = new HashMap<>();
    for( V elem : lst ) {
      T sel = selector.apply( elem );
      if( !groupedList.containsKey(sel) ) {
        groupedList.put( sel, new ArrayList<V>() );
      }
      groupedList.get(sel).add(elem);
    }

    return groupedList;
  }

  /**
   * Split a collection according to a selector.
   * @param lst The list to filter
   * @param selector A function determining whether the elements of the list should go into the first or second component.
   * @param <V> The type of the list.
   * @return A pair of lists, where the first contains elements for which selector is true, and the second contains elements for which it's false.
   */
  public static <V> Pair<List<V>,List<V>> split(final Collection<V> lst, Function<V,Boolean> selector) {
    List<V> lst1 = new ArrayList<>();
    List<V> lst2 = new ArrayList<>();
    for( V elem : lst ) {
      if( selector.apply( elem ) ) lst1.add(elem);
      else lst2.add(elem);
    }

    return Pair.makePair(lst1, lst2);
  }

  public static <V> void removeDuplicates (List<V> lst) {
    for(ListIterator<V>  it = lst.listIterator(); it.hasNext();) {
      int idx = it.nextIndex();
      V elem = it.next();
      if( idx < lst.size() - 1 &&
          lst.subList(idx+1,lst.size()-1).contains(elem) ) it.remove();
    }
  }

  /**
   * Group elements in a list together by equivalence classes
   */
  public static<V> List<Set<V>> groupByRankedEquivalence( Collection<V> lst,
                                                    Function<Pair<V,V>,Double> comparator ) {

    // Create a map of items and things they are equivalent to
    Map<V, List<V>> equivalenceMatching = new HashMap<>();
    startTrack("Making equivalence lists");
    for( V elem : lst ) {
      equivalenceMatching.put( elem, new ArrayList<V>() );
      ClassicCounter<V> candidates = new ClassicCounter<>();
      for( V elem_ : lst ) {
        if( elem.equals( elem_ ) ) continue; // don't add yourself

        double score = comparator.apply( Pair.makePair( elem, elem_ ) );
        if( score == Double.POSITIVE_INFINITY )
          equivalenceMatching.get(elem).add(elem_);
        else
          candidates.setCount(elem_, score);
      }
      if( Counters.max(candidates) > 0.0 ) {
        V elem_ = Counters.argmax(candidates);
        equivalenceMatching.get( elem ).add( elem_ );
      }
    }
    endTrack("Making equivalence lists");

    return groupByEquivalence(lst, equivalenceMatching);
  }

  public static<V> List<Set<V>> groupByEquivalence( Collection<V> lst,
                                                    Map<V, List<V>> equivalenceMatching ) {
    List<Set<V>> equivalenceClasses = new ArrayList<>();

    startTrack("Flattening into equivalence classes");
    while( equivalenceMatching.keySet().iterator().hasNext() ) {
      // Get some item
      V item = equivalenceMatching.keySet().iterator().next();
      Set<V> equivalenceClass = new HashSet<>();
      equivalenceClass.add(item);

      // Now merge everything that is equivalent to it.
      startTrack("Flattening entities equivalent to " + item );
      Queue<V> equivalentEntities = new LinkedList<>( equivalenceMatching.get(item) );
      while( equivalentEntities.size() > 0 ) {
        V entity = equivalentEntities.poll();
        // Ignore if you've already merged this entity
        if( !equivalenceMatching.containsKey( entity ) ) continue;

        // Otherwise add to the equivalence class and queue of things to
        // be processed.
        equivalenceClass.addAll( equivalenceMatching.get( entity ) );
        equivalentEntities.addAll( equivalenceMatching.get( entity ) );
        equivalenceMatching.remove( entity );
      }
      // Finally remove this item
      equivalenceMatching.remove( item );
      endTrack("Flattening entities equivalent to " + item );
      equivalenceClasses.add( equivalenceClass );
    }
    endTrack("Flattening into equivalence classes");

    return equivalenceClasses;
  }

  public static<V> boolean all(Collection<V> lst, Function<V,Boolean> fn ) {
    for( V elem : lst )
      if( !fn.apply(elem) ) return false;
    return true;
  }

  public static<V> boolean any(Collection<V> lst, Function<V,Boolean> fn ) {
    for( V elem : lst )
      if( fn.apply(elem) ) return true;
    return false;
  }

  /**
   * <p>
   *   Converts a collection (canonically, a Set) to an ordered list, such that
   *   any permutation of the collection passed in always returns the same list out.
   *   The implementation of this will order by toString(); an exception is thrown if
   *   two elements have to same toString() but are not identical.
   *   Order between identical elements remains undefined and only consistent within a
   *   single JVM run, by using {@link System#identityHashCode(Object)}.
   * </p>
   *
   * <p>
   *   Note that, since this must work consistently across processes, we cannot use
   *   {@link String#hashCode()}. Java went off the deep end with release 1.7 and decided
   *   that a String's hash code should be random across runs, and only consistent within the same
   *   JVM instance. #javafail
   * </p>
   *
   * @param unorderedSet The collection to convert to a canonical order
   * @param <E> The type of entry in the collection
   * @return A list of every element in the input, but canonically ordered.
   */
  public static <E> List<E> canonicallyOrder(Collection<E> unorderedSet) {
    Map<String, E> hashToElem = new HashMap<>();
    // Compute hashes
    for (E elem : unorderedSet) {
      String hash = CoreMapUtils.getHexKeyString(elem.toString());
      if (hashToElem.containsKey(hash)) {
        // Case: hash overlap. Disambiguate deterministically
        warn("hash collision in CollectionUtils.canonicallyOrder()");
        E elem2 = hashToElem.get(hash);
        hashToElem.remove(hash);
        assert elem2.equals(elem);
        if (System.identityHashCode(elem) < System.identityHashCode(elem2)) {
          hashToElem.put(hash + "x", elem);
          hashToElem.put(hash + "y", elem2);
        } else {
          hashToElem.put(hash + "x", elem2);
          hashToElem.put(hash + "y", elem);
        }
      } else {
        // Case: simply add the hash
        hashToElem.put(hash, elem);
      }
    }
    // Sort the hashes
    List<String> hashes = new ArrayList<>(hashToElem.keySet());
    Collections.sort(hashes);
    // Return element list
    List<E> rtn = new ArrayList<>();
    for (String hash : hashes) {
      rtn.add(hashToElem.get(hash));
    }
    return rtn;
  }

  /**
   * Compute all the permutations of a set of items.
   * @param unorderedSet The set of items to permute.
   * @param <E> The type of element in the set.
   * @return A set of lists, where each element of the return is a a unique permutation of the input.
   */
  public static <E> Set<List<E>> permutations(final Collection<E> unorderedSet) {
    if (unorderedSet.size() == 1) {
      return new HashSet<List<E>>() {{ add(new ArrayList<>(unorderedSet)); }};
    }
    Set<List<E>> rtn = new HashSet<>();
    for (E firstElement : unorderedSet) {
      List<E> others = new ArrayList<>(unorderedSet);
      others.remove(firstElement);
      for (List<E> subPermutations : permutations(others)) {
        subPermutations.add(firstElement);
        rtn.add(subPermutations);
      }
    }
    return rtn;
  }

  public static <E> Set<E> transitiveClosure(Collection<? extends Set<E>> equivalenceClasses, E seed) {
    return transitiveClosure(equivalenceClasses, seed, Integer.MAX_VALUE);
  }

  /**
   * Returns the transitive closure of a set of equivalence classes, given a seed element.
   * For example, if the equivalences are <i>( (a, b, c), (b, d), (e, f), (d, g) )</i> and the seed item is
   * <i>a</i>, then the transitive closure will be <i>(a, b, c, d, g)</i>
   * @param equivalenceClasses A collection of equivalence classes. Each element contains a set of items which are
   *                           equivalent and should be propagated in the closure computation.
   * @param seed The item to propagate the transitive closure from
   * @param maxDepth The maximum depth to follow the closure to. A depth of 0 will add nothing; 1 will add only the seed; etc.
   * @param <E> The type of the items being compared. Equal items must be .equals() to each other, but do not have
   *            to be == to each other.
   * @return A set of elements denoting the transitive closure of the equivalence classes given the seed element.
   */
  public static <E> Set<E> transitiveClosure(Collection<? extends Set<E>> equivalenceClasses, E seed, int maxDepth) {
    // Build graph
    Map<E, Set<E>> graph = new HashMap<>();
    for (Set<E> equivalenceClass : equivalenceClasses) {
      for (E item : equivalenceClass) {
        if (!graph.containsKey(item)) {
          graph.put(item, new HashSet<E>());
        }
        graph.get(item).addAll(equivalenceClass);
      }
    }
    // Search graph
    Set<E> closure = new HashSet<>();
    Stack<Pair<E,Integer>> fringe = new Stack<>();  // DFS, because why not.
    if (maxDepth > 0) {
      fringe.push(Pair.makePair(seed, 0));
      closure.add(seed);  // since we add on pushing into the fringe, we need to add the seed here
    }
    while (!fringe.isEmpty()) {
      Pair<E, Integer> node = fringe.pop();
      assert node.second < maxDepth;
      Set<E> children = graph.get(node.first);
      if (children != null && node.second < maxDepth - 1) {
        for (E child : children) {
          if (!closure.contains(child)) {
            fringe.add(Pair.makePair(child, node.second + 1));
          }
        }
        closure.addAll(children);
      }
    }
    // Return
    return closure;
  }

  /**
   * This is a bit of a weird function. Given a 2D array, take the cross product of each element
   * such that the first indices don't match.
   * The canonical use case is generating negative examples for a clustering task.
   * Given a set of clusters, take all pairs of elements such that they are not in the same
   * cluster.
   *
   * @param elements The elements (e.g., clusters).
   * @param condition The condition on which to take this pair; otherwise, skip over it.
   * @param <F> The type of element we are comparing.
   * @return An iterator over pairs of elements, such that they are not in the same array (cluster),
   *         and they match the condition.
   */
  public static <F> Iterator<Pair<F, F>> takePairs(final F[][] elements, final Function<Pair<F,F>, Boolean> condition) {
    return iteratorFromMaybeFactory(new Factory<Maybe<Pair<F, F>>>() {
      int[] indices = {0, 0, 0, -1};
      private boolean count() {
        if (indices[3] == -1) { indices[3] = 0; return true; }
        indices[3] = (indices[3] + 1) % elements[indices[2]].length;
        if (indices[3] == 0) {
          indices[2] = (indices[2] + 1) % elements.length;
          if (indices[2] == 0) {
            indices[1] = (indices[1] + 1) % elements[indices[0]].length;
            indices[2] = (indices[0]+1) % elements.length;
            if(indices[2] == 0) {
              return false;
            }
            if (indices[1] == 0) {
              indices[0] = (indices[0] + 1) % elements.length;

              if (indices[0] == 0 || indices[2] ==0) {
                return false;
              }
            }
          }
        }
        return true;
      }

      @Override
      public Maybe<Pair<F, F>> create() {
        if (!count()) { return null; }
        while (indices[0] == indices[2] ||
            !condition.apply(Pair.makePair(elements[indices[0]][indices[1]], elements[indices[2]][indices[3]]))) {
          if (!count()) { return null; }
        }
        return Maybe.Just(Pair.makePair(elements[indices[0]][indices[1]], elements[indices[2]][indices[3]]));
      }
    });
  }

  /**
   * Split an iterator into chunks -- this is useful for, e.g., splitting work among threads when it
   * input iterator has very short jobs.
   *
   * @param input The input iterator to chunk.
   * @param chunkSize The number of elements to include in each chunk (except the last one).
   * @param <E> The element type of the input iterator.
   * @return An iterator corresponding to the input, but chunked into groups.
   */
  public static <E> Iterator<Collection<E>> chunk(final Iterator<E> input, final int chunkSize) {
    return new Iterator<Collection<E>>() {
      @Override
      public synchronized boolean hasNext() {
        return input.hasNext();
      }

      @Override
      public synchronized Collection<E> next() {
        List<E> rtn = new ArrayList<>();
        for (int i = 0; i < chunkSize; ++i) {
          if (input.hasNext()) {
            rtn.add(input.next());
          }
        }
        return rtn;
      }

      @Override
      public void remove() {
        throw new RuntimeException("Remove is not implemented");
      }
    };
  }

  /**
   * Buffer an iterator, so that a thread is continuously reading ahead a given number of elements.
   * This is useful for, e.g., cases where we are interleaving heavy IO and heavy computation.
   * A thread can read ahead from disk while the main program is performing the computation.
   *
   * <p>WARNING: in preliminary tests, this seems to slow down with time, while taking up astronomical CPU time.</p>
   *
   * @param input The input iterator to read ahead from.
   * @param maxBufferSize The maximum number of elements to read ahead; that is, the maximum buffer size.
   * @param <E> The type of element being returned.
   * @return A buffered iterator, corresponding exactly to the input iterator but with read-ahead.
   */
  public static <E> IterableIterator<E> buffer(final Iterator<E> input, int maxBufferSize) {
    final BlockingQueue<Maybe<E>> buffer = new ArrayBlockingQueue<>(maxBufferSize);
    // Producer
    Thread producer = new Thread() {
      @Override
      public void run() {
        // Slurp input
        while (input.hasNext()) {
          try {
            buffer.put(Maybe.Just(input.next()));
          } catch (InterruptedException e) {
            e.printStackTrace();
          }
        }
        // Add last element
        try {
          buffer.put(Maybe.<E>Nothing());
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
      }
    };
    producer.setDaemon(true);
    producer.start();
    // Consumer
    return new IterableIterator<>(new Iterator<E>() {
      private E next = null;
      private boolean primed = false;
      @Override
      public synchronized boolean hasNext() {
        if (!primed) {
          try {
            next = buffer.take().orNull();
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
          primed = true;
        }
        return next != null;
      }
      @Override
      public synchronized E next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        primed = false;
        return next;
      }
      @Override
      public void remove() {
        throw new RuntimeException("remove() is no longer available in a buffered iterator");
      }
    });
  }

}
