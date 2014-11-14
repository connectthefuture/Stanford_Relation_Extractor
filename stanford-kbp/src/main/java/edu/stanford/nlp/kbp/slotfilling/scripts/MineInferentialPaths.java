package edu.stanford.nlp.kbp.slotfilling.scripts;

import edu.stanford.nlp.kbp.entitylinking.EntityLinker;
import edu.stanford.nlp.kbp.entitylinking.WikidictEntityLinker;
import edu.stanford.nlp.kbp.slotfilling.SlotfillingSystem;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KnowledgeBase;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;
import java.util.function.Function;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Mine short inferential paths using reverb and relation classifier produced graphs.
 */
public class MineInferentialPaths {
  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Miner");

  @Execution.Option(name="mine-inferential-paths.begin", required=true, gloss="The query entity to begin with")
  private static int begin = 0;
  @Execution.Option(name="mine-inferential-paths.count", required=true, gloss="The number of entities to query over")
  private static int count = Integer.MAX_VALUE;
  @Execution.Option(name="mine-inferential-paths.cutoff", gloss="Only take KBP relations not in the kB above this threshold")
  private static double cutoff = 0.0;


  public static void main(String[] args) {
    SlotfillingSystem.exec(in -> {
      // Use a very high-precision entity linker
      try {
        Props.ENTITYLINKING_LINKER = Lazy.<EntityLinker>from(new WikidictEntityLinker());
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      // Create our system
      SlotfillingSystem system = new SlotfillingSystem(in);
      final InferentialSlotFiller slotFiller = new InferentialSlotFiller(in, system.getIR(),
              system.getProcess(), system.getTrainedClassifier().get(),
          new GoldResponseSet());
      final KBPIR ir = system.getIR();

      // Get Queries
      final KnowledgeBase kb = system.getIR().getKnowledgeBase();
      TreeSet<KBPOfficialEntity> entitySet = new TreeSet<>();
      for (KBTriple triple : kb.triples()) {
        entitySet.add(KBPNew.from(triple.getEntity()).KBPOfficialEntity());
      }
      final KBPOfficialEntity[] entities = entitySet.toArray(new KBPOfficialEntity[entitySet.size()]);
      logger.log(BLUE, "" + entities.length + " entities in KB");
      logger.log(BLUE, "querying [" + begin + ", " + Math.min(begin + count, entities.length) + "]");

      // Annotate files
      PostgresUtils.withSet("mined_documents", new PostgresUtils.SetCallback() {
        @Override
        public void apply(final Connection psql) throws SQLException {
          try {
            // CASE: Run Query By Query
            for (int i = begin; i < Math.min(begin + count, entities.length); ++i) {
              KBPOfficialEntity pivot = entities[i];
              forceTrack("Mining paths for " + pivot);
              try {
                final Set<String> docidsToRegister = new HashSet<>();
                runOnGraph(
                    enforceKBInGraph(
                        slotFiller.extractRelationGraph(pivot, Props.TEST_SENTENCES_PER_ENTITY, Maybe.Just(new Function<String, Boolean>() {
                  @Override
                  public Boolean apply(String docid) {
                    try {
                      if (contains(psql, "mined_documents", docid)) {
                        return false;
                      }
                      docidsToRegister.add(docid);
                    } catch (SQLException e) {
                      logger.err(e);
                    }
                    return true;
                  }
                })), kb), ir);
                for (String docid : docidsToRegister) { add(psql, "mined_documents", docid); }
                flush(psql, "mined_documents");
              } catch (Throwable e) {
                logger.err(e);
              } finally {
                endTracksUntil("Mining paths for " + pivot);
              }
              endTrack("Mining paths for " + pivot);

            }
          } catch (Exception e) {
            logger.err(e);
          }
        }
      });

      return null;
    }, args);
  }

  /**
   * Enforce that the relations in the graph are consistent with the KB relations (in a trivial sense),
   * and that all the edges in the KB are in the graph.
   * @param graph The graph to augment and check for consistency
   * @param kb The knowledge base
   * @return The same graph, but mutated so that it is consistent with the knowledge base
   */
  private static EntityGraph enforceKBInGraph(EntityGraph graph, KnowledgeBase kb) {
    startTrack("Syncing with KB");
    // Filter contradictory relations
    Iterator<KBPSlotFill> iter = graph.edgeIterator();
    OUTER: while (iter.hasNext()) {
      // variables
      KBPSlotFill fill = iter.next();
      KBPEntity source = fill.key.getEntity();
      // Check
      if (kb.data.containsKey(source)) {  // if entity in KB
        LinkedHashSet<KBPSlotFill> knownEdges = kb.data.get(source);
        for (KBPSlotFill knownEdge : knownEdges) {  // for each known relation in the KB
          if (knownEdge.key.slotValue.equals(fill.key.slotValue)) {  // if the slot values also match
            for (RelationType guessRel : fill.key.tryKbpRelation()) {  // if it's an official relation
              for (RelationType goldRel : knownEdge.key.tryKbpRelation()) {  // if the KB element is an official relation
                if (!goldRel.plausiblyCooccursWith(guessRel)) {  // if the two relations don't plausibly co-occur
                  logger.log("Filtered impossible relation: " + guessRel + " on account of " + guessRel);
                  iter.remove();  // remove it!
                  continue OUTER;
                }
              }
            }
          }
        }
      }
    }  // wow, so many braces!

    // Construct slot name to entity map
    // This is because the KB drops the entity type :/
    Map<String, KBPEntity> nameToEntity = new HashMap<>();
    for (KBPEntity entity : graph.getAllVertices()) {
      nameToEntity.put(entity.name, entity);
    }

    // Filter out inferred KBP entities, if applicable
    if (cutoff > 0.0) {
      Iterator<KBPSlotFill> edges = graph.edgeIterable().iterator();
      while (edges.hasNext()) { if (edges.next().score.getOrElse(1.0) < cutoff) { edges.remove(); } }
    }

    // Add slot values in KB not already in the graph
    for (KBPEntity entity : kb.data.keySet()) {  // iterate over KB
      if (graph.containsVertex(entity)) {  // if graph has entity
        for (KBPSlotFill trueRelation : kb.data.get(entity)) {  // for each true relation over this entity
          KBPEntity target = nameToEntity.get(trueRelation.key.slotValue);
          if (target == null) { continue; }
          trueRelation = KBPNew.from(trueRelation).slotValue(target).score(1.0).KBPSlotFill();
          if (graph.containsVertex(target)) {  // if the slot value is in the graph as well
            List<KBPSlotFill> fills = graph.getEdges(entity, target);
            if (!fills.contains(trueRelation)) {  // if it's not already added to the graph
              logger.log("Adding known relation: " + trueRelation);
              graph.add(entity, target, trueRelation);  // add it to the graph
            }
          }
        }
      }
    }

    // Return the graph
    endTrack("Syncing with KB");
    return graph;
  }

  public static void runOnGraph(EntityGraph graph, KBPIR ir) {
    // Run Filters
    graph = new GraphConsistencyPostProcessors.UnaryConsistencyPostProcessor(SlotfillPostProcessor.unary).postProcess(graph);

    // vv Extract Paths vv
    Counter<List<KBTriple>> preds = extractAllFormulas(graph, ir);
    // ^^               ^^

    // Save path
    logger.log(GREEN, "" + preds.size() + " formulas extracted");
    saveInferentialPaths(preds);
    // Print path
    if (preds.size() <= 100) {
      startTrack("Formulas");
      for( Map.Entry<List<KBTriple>, Double> path : preds.entrySet()) {
        logger.log(path.getValue() + ": " + StringUtils.join(path.getKey(), " ∧ "));
      }
      endTrack("Formulas");
    }
  }

  /**
   * Saves an inferential path to Postgres
   */
  private static void saveInferentialPaths(final Counter<List<KBTriple>> preds) {
    PostgresUtils.withCounter(Props.DB_TABLE_MINED_FORMULAS, new PostgresUtils.CNFFormulaCounterCallback() {
      @Override
      public void apply(Connection psql) throws SQLException {
        for( Map.Entry<List<KBTriple>, Double> pred : preds.entrySet()) {
          // Write
          if (Utils.doesLoop(pred.getKey())) {
            assert Utils.doesLoop(string2key(key2string(pred.getKey())));
            List<KBTriple> antecedent = pred.getKey().subList(0, pred.getKey().size() - 1);
            assert preds.getCount(string2key(key2string(antecedent))) > 0.0;
          }
          incrementCount(psql, Props.DB_TABLE_MINED_FORMULAS, pred.getKey(), pred.getValue());
        }
        flush(psql, Props.DB_TABLE_MINED_FORMULAS);
      }
    });
  }

  /** A simple enum for whether an edge goes "forwards" or "backwards" in a path */
  public static enum Direction { FORWARD, BACKWARD }

  /**
   * A simple Trie class, representing a (partial) path through a graph.
   */
  public static class Trie {
    /** The entity at the end of the path */
    public final KBPEntity entry;
    /** The children from this entity */
    public final Map<Triple<String,Direction,KBPEntity>, Trie> children;
    /** The relation and direction of the relation of the parent of this entity in the path */
    public final Pair<String, Direction> relationFromParent;
    /** The parent of this entity, as a Trie in itself */
    public final Trie parent;

    /** Create a new Trie, rooted at this entity with no outgoing edges */
    protected Trie(KBPEntity entry) {
      this.entry = entry;
      this.children = new HashMap<>();
      this.relationFromParent = null;
      this.parent = null;
    }

    /** Create a new Trie from an incoming edge */
    protected Trie(Triple<String, Direction, KBPEntity> incomingEdge, Trie parent) {
      this.entry = incomingEdge.third;
      this.children = new HashMap<>();
      this.relationFromParent = Pair.makePair(incomingEdge.first, incomingEdge.second);
      this.parent = parent;
    }

    /** The depth of the Trie, as counted by the number of edges the path this Trie represents contains */
    public int depth() {
      if (parent == null) { return 0; } else { return 1 + parent.depth(); }
    }

    /** The root entity of this Trie */
    public KBPEntity root() {
      if (parent == null) { return entry; }
      return parent.root();
    }

    /** Return whether this Trie represents a complete loop */
    public boolean isLoop() {
      return parent != null && root().equals(entry);
    }

    /**
     * Returns whether extending this Trie with the given entity would create a loop over a part of the Trie,
     * but not a complete loop.
     * For example, A-&gt;B-&gt;C-&gt;B is a dangling loop.
     */
    public boolean danglingLoop(KBPEntity entity) {
      return parent != null && (entry.equals(entity) || parent.danglingLoop(entity));
    }

    /**
     * Extend this Trie with an edge, computing the direction automatically.
     *
     * @param triple The edge to add; note that at least one end of the edge must be equal to the entity in
     *               this Trie.
     *
     * @return Optionally return the child Trie, if we should continue to extend this path, or {@link edu.stanford.nlp.kbp.common.Maybe#Nothing} if the resulting path should not be extended.
     */
    public Maybe<Trie> extend(KBTriple triple) {
      KBPEntity entity = triple.getEntity();
      KBPEntity slotValue = triple.getSlotEntity().orCrash();
      Triple<String, Direction, KBPEntity> key;
      if (entity.equals(entry) || (!slotValue.equals(entry) && entity.name.equals(entry.name))) {
        if (danglingLoop(slotValue)) {
          return Maybe.Nothing();  // this would cause a dangling loop
        }
        if (parent != null && parent.entry.equals(slotValue) && relationFromParent.first.equals(triple.relationName) && relationFromParent.second == Direction.BACKWARD) {
          return Maybe.Nothing();  // we're trivially backtracking
        }
        key = Triple.makeTriple(triple.relationName, Direction.FORWARD, slotValue);
      } else if (slotValue.equals(entry) || slotValue.name.equals(entry.name)) {
        if (danglingLoop(entity)) {
          return Maybe.Nothing();  // this would cause a dangling loop
        }
        if (parent != null && parent.entry.equals(entity) && relationFromParent.first.equals(triple.relationName) && relationFromParent.second == Direction.FORWARD) {
          return Maybe.Nothing();  // we're trivially backtracking
        }
        key = Triple.makeTriple(triple.relationName, Direction.BACKWARD, triple.getEntity());
      } else {
        throw new IllegalStateException("Cannot add edge to Trie: " + triple + "; Trie ends at " + this.entry);
      }

      if (children.containsKey(key)) {
        return Maybe.Nothing(); // this edge already exists
      } else {
        Trie value = new Trie(key, this);
        if (value.isLoop()) {
          // Add all loops
          children.put(key, value);
          return Maybe.Nothing();
        } else if (value.depth() <= Props.TEST_GRAPH_INFERENCE_DEPTH) {
          // Add anything under the max depth
          children.put(key, value);
          return Maybe.Just(value);
        } else {
          // Don't add anything else -- unless it's a loop, covered above
          return Maybe.Nothing();
        }
      }
    }

    /** Returns this Trie as a path, looking <b>backwards</b> but not forwards */
    public List<KBTriple> asPath() {
      if (parent == null) {
        return new ArrayList<>();
      } else {
        List<KBTriple> path = parent.asPath();
        KBPEntity source = relationFromParent.second == Direction.FORWARD ? parent.entry : entry;
        KBPEntity target = relationFromParent.second == Direction.BACKWARD ? parent.entry : entry;
        path.add(KBPNew.from(source).slotValue(target).rel(relationFromParent.first).KBTriple());
        return path;
      }
    }

    /**
     * Enumerate all paths in this Trie. This should be called from the Trie's root, or else it will
     * return all paths in the Trie which have this Trie as a prefix.
     */
    public Collection<List<KBTriple>> allPathsInTrie() {
      List<List<KBTriple>> paths = new ArrayList<>();
      if (parent != null) {
        paths.add(asPath());
      }
      for (Trie child : children.values()) {
        paths.addAll(child.allPathsInTrie());
      }
      return paths;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Trie)) return false;
      Trie trie = (Trie) o;
      return entry.equals(trie.entry) && parent.equals(trie.parent) && relationFromParent.equals(trie.relationFromParent);

    }

    @Override
    public int hashCode() {
      int result = entry.hashCode();
      result = 31 * result + relationFromParent.hashCode();
      return result;
    }

    private String toString(String indent) {
      String rtn = entry + "\n";
      for (Trie child : children.values()) {
        rtn += indent + (child.relationFromParent.second == Direction.FORWARD ? "  -["+child.relationFromParent.first+"]-> " : "  <-["+child.relationFromParent.first+"]- ")
            + child.toString(indent + "    ");
      }
      return rtn;
    }
    @Override
    public String toString() {
      return toString("");
    }
  }


  /**
   * Extract all paths of up-to maximum depth maxDepth.
   *
   * <p>
   *   Note that a clause instance has weight:
   *     1 / (# documents with all entities in the clause),
   *     potentially approximated as 1.0, which is then added to the clause template as
   *     a count.
   *   This ensures that we are not overly biasing our counts towards entities which happen to
   *     co-occur often, and are therefore more likely to have some relation a priority.
   *   Then, any path which forms a loop also has the same fractional count appended to every subset of the clause
   *   of length (n) which is of length (n-1) -- that is, the "(n-1) grams" in a sense.
   * </p>
   *
   * @param graph The entity graph to extract predicate paths on.
   * @param ir The slotfilling system's IR component, for querying document match counts.
   */
  public static Counter<List<KBTriple>> extractAllFormulas(final EntityGraph graph, final KBPIR ir) {
    List<List<KBTriple>> pathsToAbstract = new ArrayList<>();

    // Initialize Tries
    Map<KBPEntity, Trie> tries = new HashMap<>();
    Queue<Trie> fringe = new LinkedList<>();
    for (KBPEntity vertex : graph.getAllVertices()) {
      Trie trie = new Trie(vertex);
      tries.put(vertex, trie);
      fringe.add(trie);
    }

    // Run search
    while (!fringe.isEmpty()) {
      Trie leader = fringe.poll();
      for (KBPSlotFill outgoingFill : graph.outgoingEdgeIterable(leader.entry)) {
        for (Trie child : leader.extend(outgoingFill.key)) {
          fringe.add(child);
        }
      }
      for (KBPSlotFill incomingFill : graph.incomingEdgeIterable(leader.entry)) {
        for (Trie child : leader.extend(incomingFill.key)) {
          fringe.add(child);
        }
      }
    }

    for (Trie trie : tries.values()) {
      pathsToAbstract.addAll(trie.allPathsInTrie());
    }

    // Multithreading -- split data
    int numThreads = Execution.threads;
    final List<List<List<KBTriple>>> inputData = new ArrayList<>();
    for (int i = 0; i < numThreads; ++i) { inputData.add(new ArrayList<List<KBTriple>>()); }
    int i = 0;
    for (List<KBTriple> path : pathsToAbstract) {
      inputData.get(i % numThreads).add(path);
      i += 1;
    }
    // Multithreading -- create tasks
    List<Runnable> tasks = new ArrayList<>();
    final List<Counter<List<KBTriple>>> outputs = new ArrayList<>();  // threadsafe output
    for (i = 0; i < numThreads; ++i) {
      final int index = i;
      outputs.add(new ClassicCounter<List<KBTriple>>());
      tasks.add(() -> {
        Map<Set<String>, Double> docFactorCache = new HashMap<>();  // only for efficiency to save Lucene lookups -- inside thread since it's not threadsafe
        for (List<KBTriple> path : inputData.get(index)) {
          // Actually populate the paths
          // Pay attention to me! Here be math, subtly hidden in dragons, hidden in code.
          // Abstract the formula
          List<KBTriple> abstraction;
          if (Utils.doesLoop(path)) {
            abstraction = Utils.normalizeEntailment(path);
            assert Utils.doesLoop(abstraction);
          } else {
            abstraction = new ArrayList<>(Utils.normalizeConjunction(path));
          }
          // Get the number of documents with all the entities in this formula.
          double docFactor = 1.0;  // note[gabor]: enable me for fancy math: numDocumentsContaining(path, ir, docFactorCache);
          // Increment the count of this formula, weighted by the number of times we've seen the document
          // to account for bias guessing relations for entities that often occur together.
          outputs.get(index).incrementCount(abstraction, 0.5 / docFactor);
          // Note that all the antecedents should be added when they are added as shorter paths;
          // and, that the docFactor for those will be identical to the one added here, as the entities
          // will be the same.
        }
      });
    }
    // Multithreading -- run
    threadAndRun(tasks);
    // Multithreading -- merge
    final Counter<List<KBTriple>> preds = new ClassicCounter<>();
    for (Counter<List<KBTriple>> output : outputs) {
      Counters.addInPlace(preds, output);
    }

    // Return
    return preds;
  }

  /**
   * Return the number of documents containing the entities in the current graph.
   *
   * @param clauses The clauses to extract the entities from.
   * @param ir An IR system to collect counts from.
   * @param cache A cache, to avoid IR hits.
   * @return The number of documents containing the entities in the clauses given.
   */
  @SuppressWarnings("SynchronizationOnLocalVariableOrMethodParameter")
  private static double numDocumentsContaining(Collection<KBTriple> clauses, KBPIR ir, Map<Set<String>, Double> cache) {
    Set<String> terms = new HashSet<>();
    for (KBTriple clause : clauses) {
      terms.add(clause.entityName);
      terms.add(clause.slotValue);
    }
    Double cachedCount;
    cachedCount = cache.get(terms);
    if (cachedCount != null) {
      return cachedCount;
    } else {
      try {
        cachedCount = Math.max(1.0, ir.queryNumHits(terms));
      } catch (Exception e) {
        logger.err(e);
        cachedCount = 1.0;
      }
      cache.put(terms, cachedCount);
      return cachedCount;
    }
  }

}
