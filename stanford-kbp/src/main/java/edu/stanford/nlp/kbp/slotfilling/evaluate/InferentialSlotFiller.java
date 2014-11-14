package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.graph.GraphAlgorithms;
import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.kbp.slotfilling.classify.OpenIERelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.inference.GraphInferenceEngine;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.kbp.slotfilling.process.*;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess.AnnotateMode;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;
import java.util.function.Function;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * An extension of the slot filler that uses some graph inference
 */
public class InferentialSlotFiller extends SimpleSlotFiller {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Infer");
  protected OpenIERelationExtractor reverbRelationExtractor;
  protected GraphInferenceEngine graphInferenceEngine;

  public InferentialSlotFiller(Properties props,
                               KBPIR ir,
                               KBPProcess process,
                               RelationClassifier classify,
                               GoldResponseSet goldResponses
  ) {
    super( props, ir, process, classify, goldResponses );
    reverbRelationExtractor = MetaClass.create(Props.TEST_GRAPH_OPENIE_CLASS).createInstance();
    if( Props.TEST_GRAPH_INFERENCE_DO ) {
      try {
        graphInferenceEngine = MetaClass.create(Props.TEST_GRAPH_INFERENCE_CLASS).createInstance(goldResponses);
      } catch (MetaClass.ConstructorNotFoundException e) {
        graphInferenceEngine = MetaClass.create(Props.TEST_GRAPH_INFERENCE_CLASS).createInstance();
      }
    }
  }

  @Override
  public List<KBPSlotFill> fillSlots(final KBPOfficialEntity queryEntity) {
    startTrack(BLUE, BOLD, "Annotating " + queryEntity);

    // -- Raw Classification
    startTrack("Raw Classification");
    // -- Construct Graph
    // Read slot candidates from the index
    // Each of these tuples effectively represents (e_1, e_2, [datums]) where e_1 is the query entity.
    EntityGraph relationGraph = extractRelationGraph(queryEntity, Props.TEST_SENTENCES_PER_ENTITY, Maybe.<Function<String, Boolean>>Nothing());
    assert relationGraph.containsVertex(queryEntity);
    if (!Props.TEST_GRAPH_KBP_DO) {
      Iterator<KBPSlotFill> edges = relationGraph.edgeIterator();
      while (edges.hasNext()) {
        if (edges.next().key.hasKBPRelation()) { edges.remove(); }
      }
    }

    // Prune the graph to vertices within a distance of 4 (We don't do inference any more than that) from the query entity.
    Set<KBPEntity> closeEntities = GraphAlgorithms.getDistances(relationGraph, queryEntity, 4).keySet();
    List<KBPEntity> entities = new ArrayList<>(relationGraph.getAllVertices());
    for(KBPEntity entity : entities) {
      if(!closeEntities.contains(entity)) relationGraph.removeVertex(entity);
    }

    assert relationGraph.containsVertex(queryEntity);
    logger.log( relationGraph.getNumVertices() + " entities in the graph with " + relationGraph.getNumEdges() + " slot fills" );
    logger.log(relationGraph.getOutDegree(queryEntity) + " official slot fills remain in graph");
    endTrack("Raw Classification");

    // -- Run a first pass of consistency
    startTrack("Local Consistency (run 1)");
    // Go about clearing each of the edges.
    relationGraph = new GraphConsistencyPostProcessors.UnaryConsistencyPostProcessor(queryEntity, SlotfillPostProcessor.unary(irComponent)).postProcess(relationGraph, goldResponses);
    logger.log(relationGraph.getNumVertices() + " entities in the graph with " + relationGraph.getNumEdges() + " slot fills");
    logger.log( relationGraph.getOutDegree(queryEntity) + " official slot fills remain in graph" );
    endTrack("Local Consistency (run 1)");

    // -- Run Simple Inference
    startTrack("Inference");
    if (Props.TEST_RELATIONFILTER_DO) {
      relationGraph = new GraphConsistencyPostProcessors.SentenceCompetitionPostProcessor().postProcess(relationGraph, goldResponses);
    }
    // Merge vertices on the graph
    if( Props.TEST_GRAPH_MERGE_DO ) {
      relationGraph = new GraphConsistencyPostProcessors.EntityMergingPostProcessor().postProcess(relationGraph, goldResponses);
      assert relationGraph.containsVertex(queryEntity);
    }
    // Compute transitive completion of relations
    if (Props.TEST_GRAPH_TRANSITIVE_DO) {
      relationGraph = new GraphConsistencyPostProcessors.TransitiveRelationPostProcessor().postProcess(relationGraph, goldResponses);
    }
    // Symmetric function rewrite by default
    if (Props.TEST_GRAPH_SYMMETERIZE_DO) {
      relationGraph = new GraphConsistencyPostProcessors.SymmetricFunctionRewritePostProcessor().postProcess(relationGraph, goldResponses);
    }

    List<KBPSlotFill> initiallyDiscardedEdges = new ArrayList<>();
    if(Props.TEST_GRAPH_INFERENCE_HACKS_GLOBAL_CONSISTENCY) {
      startTrack("Global Consistency (run 2)");
      // Go about adding provenance and removing any official entity that doesn't have it.
      List<KBPSlotFill> kbpEdges = CollectionUtils.filter(relationGraph.getOutgoingEdges(queryEntity), in -> in.key.hasKBPRelation());
      // Now remove these edges from the graph
      for(KBPSlotFill fill : kbpEdges) {
        //noinspection AssertWithSideEffects
        assert fill.key.getEntity().equals(queryEntity);
        relationGraph.removeEdge(queryEntity, fill.key.getSlotEntity().get(), fill);
      }
      // Book-keeping
      initiallyDiscardedEdges.addAll(kbpEdges);

      // Run kbpEdges through consistency.
      List<KBPSlotFill> withProvenance = new ArrayList<>();
      for (KBPSlotFill fill : kbpEdges) {
        KBPSlotFill augmented = KBPNew.from(fill).provenance(findBestProvenance(queryEntity, fill)).KBPSlotFill();
        if (augmented.provenance.isDefined() && (!Props.TEST_PROVENANCE_DO || augmented.provenance.get().isOfficial())) {
          withProvenance.add(augmented);
        } else {
          initiallyDiscardedEdges.add(fill);
          goldResponses.discardNoProvenance(fill);
        }
      }
      // Run consistency pass 2
      List<KBPSlotFill> consistentRelations = this.finalConsistencyAndProvenancePass(queryEntity, withProvenance, goldResponses);
      initiallyDiscardedEdges.removeAll(consistentRelations);

      // Now put this back into the graph.
      for(KBPSlotFill fill : consistentRelations) {
        relationGraph.add(queryEntity, fill.key.getSlotEntity().get(), fill);
      }

      logger.log(relationGraph.getNumVertices() + " entities in the graph with " + relationGraph.getNumEdges() + " slot fills");
      logger.log( relationGraph.getOutDegree(queryEntity) + " official slot fills remain in graph" );
      endTrack("Global Consistency (run 2)");
    }

    // -- Add Inference Rules
    if( Props.TEST_GRAPH_INFERENCE_DO && graphInferenceEngine != null ) {
      forceTrack("Rules Inference");
      relationGraph = graphInferenceEngine.apply(relationGraph, queryEntity);
      endTrack("Rules Inference");
    }
    logger.log( relationGraph.getNumVertices() + " entities in the graph with " + relationGraph.getNumEdges() + " slot fills" );
    logger.log( relationGraph.getOutDegree(queryEntity) + " official slot fills remain in graph" );
    // From now on, it's no more graph, all just the query entity
    assert relationGraph.containsVertex(queryEntity);
    assert relationGraph.isValidGraph();
    List<KBPSlotFill> cleanRelations = CollectionUtils.filter(relationGraph.getOutgoingEdges(queryEntity), in -> in.key.hasKBPRelation());
    logger.log("" + cleanRelations.size() + " slot fills remain at end of inference");

    // Make up for any re-inferred slots
    for(KBPSlotFill discarded : initiallyDiscardedEdges) {
      if(cleanRelations.contains(discarded)) {
        goldResponses.undoDiscard(discarded);
        goldResponses.registerResponse(discarded);
      }
    }
    endTrack("Inference");

    // -- Run final consistency checks
    startTrack("Full Consistency + Provenance");
    // Find missing provenances
    List<KBPSlotFill> withProvenance;
    withProvenance = new ArrayList<>(cleanRelations.size());
    for (KBPSlotFill fill : cleanRelations) {
      KBPSlotFill augmented = KBPNew.from(fill).provenance(findBestProvenance(queryEntity, fill)).KBPSlotFill();
      if (augmented.provenance.isDefined() && (!Props.TEST_PROVENANCE_DO || augmented.provenance.get().isOfficial())) {
        withProvenance.add(augmented);
      } else {
        goldResponses.discardNoProvenance(fill);
      }
    }
    for (KBPSlotFill slot : withProvenance) { goldResponses.registerResponse(slot); } // re-register after provenance
    logger.log("" + withProvenance.size() + " slot fills remain after provenance");
    // Run consistency pass 2
    List<KBPSlotFill> consistentRelations = this.finalConsistencyAndProvenancePass(queryEntity, withProvenance, goldResponses);
    //List<KBPSlotFill> consistentRelations = withProvenance;
    endTrack("Full Consistency + Provenance");


    // -- Print Judgements
    logger.log("Memory usage: " + Utils.getMemoryUsage());
    goldResponses.appendForEntity(queryEntity, Maybe.Just(irComponent));
    logger.prettyLog(goldResponses.loggableForEntity(queryEntity, Maybe.Just(irComponent)));

    endTrack("Annotating " + queryEntity);

    return consistentRelations;
  }

  //
  // Public Utilities
  @SuppressWarnings("unchecked")
  public EntityGraph extractRelationGraph(final KBPOfficialEntity entity,
                                          int documentsPerEntity,
                                          Maybe<? extends Function<String, Boolean>> docidFilter) {
    // -- Check Cache
    if (Props.CACHE_GRAPH_DO && !Props.CACHE_GRAPH_REDO) {
      final Pointer<EntityGraph> cachedGraph = new Pointer<>();
      PostgresUtils.withKeyGraphTable(Props.DB_TABLE_GRAPH_CACHE, new PostgresUtils.KeyGraphCallback() {
        @Override
        public void apply(Connection psql) throws SQLException {
          cachedGraph.set(get(psql, Props.DB_TABLE_GRAPH_CACHE, keyAndGraphPropertiesToString((entity))));
        }
      });
      if (cachedGraph.dereference().isDefined()) {
        logger.log("found graph in cache");
        assert cachedGraph.dereference().get().containsVertex(entity);
        return cachedGraph.dereference().get();
      }
    }

    // -- IR
    // Get supporting sentences
    // Get sentences number of documents and keep pulling out sentences
    // till you find sufficiently many of them.
    forceTrack("Querying IR");
    List<Annotation> rawDocuments = new ArrayList<>();
    if (docidFilter.isDefined()) {
      // We don't want to take all the documents, so we query docids first
      List<String> docids = irComponent.queryDocIDs(entity.name, entity.type,
          entity.representativeDocumentId().isDefined() ? new HashSet<>(Arrays.asList(entity.representativeDocumentId().get())) : new HashSet<String>(),
          documentsPerEntity * 5);
      Iterator<String> docidIter = docids.iterator();
      while (rawDocuments.size() < documentsPerEntity && docidIter.hasNext()) {
        String docid = docidIter.next();
        if (!docidFilter.isDefined() || docidFilter.get().apply(docid)) { rawDocuments.add(irComponent.fetchDocument(docid)); }
      }
    } else {
      // This is the standard case, where we query documents directly
      rawDocuments.addAll(irComponent.queryDocuments(entity,
          entity.representativeDocumentId().isDefined() ? new HashSet<>(Arrays.asList(entity.representativeDocumentId().get())) : new HashSet<String>(),
          documentsPerEntity));
    }
    logger.log("fetched " + rawDocuments.size() + " documents");
    endTrack("Querying IR");
    if( rawDocuments.size() == 0 ) {
      logger.warn("No documents found :-/!");
      return new EntityGraph();
    }

    // Begin constructing the graph

    // -- First pass construct a graph using the relation classifier
    forceTrack("Constructing graph");
    final EntityGraph graph = extractRelationGraphFromSimpleSlotFiller(entity, rawDocuments);
    endTrack("Constructing Graph");
    assert graph.containsVertex(entity);
    graph.restrictGraph(graph.getConnectedComponent(entity));
    assert graph.containsVertex(entity);
    logger.log(BOLD, "Num Edges: " + graph.getNumEdges() );
    logger.log("Memory usage: " + Utils.getMemoryUsage() );

    // -- Second pass for Reverb
    if( Props.TEST_GRAPH_OPENIE_DO) {
      // Take care of the annotation required
      AnnotationPipeline pipeline = new AnnotationPipeline();
      pipeline.addAnnotator(new PostIRAnnotator(entity, true));
      pipeline.addAnnotator(new EntityMentionAnnotator(entity));
      pipeline.addAnnotator(new SlotMentionAnnotator());
      pipeline.addAnnotator(new RelationMentionAnnotator(entity, Collections.EMPTY_LIST, AnnotateMode.ALL_PAIRS));

      forceTrack("Augmenting with Reverb extractions");
      for( Annotation doc : rawDocuments ) {
        // Annotate with the details
        doc = CoreMapUtils.copyDocument(doc);
        pipeline.annotate(doc);
        // Extract relations using ReVerb
        for( KBPSlotFill fill :  reverbRelationExtractor.extractRelations(doc) ) {
          if( (!Props.TEST_GRAPH_OPENIE_PRUNE || graphInferenceEngine.isUsefulRelation(fill.key.relationName)) && fill.key.getSlotEntity().isDefined() ) {
            graph.add( fill.key.getEntity(), fill.key.getSlotEntity().get(), fill );
          }
        }
      }
      endTrack("Augmenting with Reverb extractions");
    }
    logger.log(BOLD, "Num Edges: " + graph.getNumEdges() );
    logger.log("Memory usage: " + Utils.getMemoryUsage());

    // -- Update Cache
    if (Props.CACHE_GRAPH_DO) {
      PostgresUtils.withKeyGraphTable(Props.DB_TABLE_GRAPH_CACHE, new PostgresUtils.KeyGraphCallback() {
        @Override
        public void apply(Connection psql) throws SQLException {
          put(psql, Props.DB_TABLE_GRAPH_CACHE, keyAndGraphPropertiesToString(entity), graph);
        }
      });
    }
    return graph;
  }

  @SuppressWarnings("unchecked")
  protected EntityGraph extractRelationGraphFromSimpleSlotFiller(KBPEntity originalQueryEntity, List<Annotation> documents) {
    EntityGraph graph = new EntityGraph();

    // Run necessary annotators
    AnnotationPipeline initialPipeline = new AnnotationPipeline();
    initialPipeline.addAnnotator(new PostIRAnnotator(originalQueryEntity instanceof KBPOfficialEntity ? (KBPOfficialEntity) originalQueryEntity : KBPNew.from(originalQueryEntity).KBPOfficialEntity(), true));
    initialPipeline.addAnnotator(new EntityMentionAnnotator(originalQueryEntity));
    initialPipeline.addAnnotator(new SlotMentionAnnotator());
    for (Annotation doc : documents) { initialPipeline.annotate(doc); }

    // Get all entities in documents
    List<Set<KBPEntity>> cooccurrences = new ArrayList<>();
    for (Annotation doc : documents) {
      for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
        // Get mentions
        Set<EntityMention> mentions = new HashSet<>();
        mentions.addAll(Maybe.fromNull(sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class)).getOrElse(Collections.EMPTY_LIST));
        mentions.addAll(Maybe.fromNull(sentence.get(KBPAnnotations.SlotMentionsAnnotation.class)).getOrElse(Collections.EMPTY_LIST));
        // Compute co-occurrences
        Set<KBPEntity> cooccurrenceSet = new HashSet<>();
        for (EntityMention mention : mentions) {
          for (NERTag mentionType : Utils.getNERTag(mention)) {
            KBPEntity candidate = KBPNew.entName(mention.getNormalizedName() != null ? (Utils.isInteger(mention.getNormalizedName()) ? mention.getFullValue() : mention.getNormalizedName()) : mention.getFullValue()).entType(mentionType).KBPEntity();
            if (candidate.equals(originalQueryEntity)) {
              cooccurrenceSet.add(originalQueryEntity);
            } else {
              cooccurrenceSet.add(candidate);
            }
          }
        }
        cooccurrences.add(cooccurrenceSet);
      }
    }
    Set<KBPEntity> allEntities = CollectionUtils.transitiveClosure(cooccurrences, originalQueryEntity, Props.TEST_GRAPH_DEPTH);

    // Some debug output
    assert allEntities.contains(originalQueryEntity);
    int totalPivotCount = 0;
    for (KBPEntity entity : allEntities) { if (entity.type.isEntityType()) { totalPivotCount += 1; } }
    logger.log("" + allEntities.size() + " potential graph vertices; " + totalPivotCount + " in entity position");

    // Classify for each entity
    for (KBPEntity pivot : allEntities) {
      assert pivot == originalQueryEntity || !pivot.equals(originalQueryEntity);
      graph.addVertex(pivot);
      if (!pivot.type.isEntityType()) { continue; }
      startTrack("Augmenting " + pivot);
      // Get relevant sentences
      List<CoreMap> relevantSentences = new ArrayList<>();
      for (Annotation doc : documents) {
        for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
          // Clear existing annotations
          sentence.remove(MachineReadingAnnotations.EntityMentionsAnnotation.class);
          sentence.remove(KBPAnnotations.SlotMentionsAnnotation.class);
          sentence.remove(MachineReadingAnnotations.RelationMentionsAnnotation.class);
          // Check if sentence is relevant
          Set<String> antecedents = sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class);
          if (antecedents.contains(pivot.name)) {
            relevantSentences.add(sentence);
          }
        }
      }
      // Get datums
      Annotation annotation = new Annotation("");
      annotation.set(CoreAnnotations.SentencesAnnotation.class, process.annotateSentenceFeatures(pivot, relevantSentences, AnnotateMode.NORMAL));
      Map<KBPair, Pair<SentenceGroup, List<CoreMap>>> datums = process.featurizeWithSentences(annotation, Maybe.<RelationFilter>Nothing());
      List<SentenceGroup> groups = new ArrayList<>();
      Map<KBPair, CoreMap[]> sentences = new HashMap<>();
      for (Map.Entry<KBPair, Pair<SentenceGroup, List<CoreMap>>> datum : datums.entrySet()) {
        if (datum.getKey().getEntity().equals(pivot)) {
          groups.add( Props.HACKS_DISALLOW_DUPLICATE_DATUMS ? datum.getValue().first.removeDuplicateDatums() : datum.getValue().first );
          sentences.put(datum.getKey(), datum.getValue().second.toArray(new CoreMap[datum.getValue().second.size()]));
        }
      }
      // Classify
      List<KBPSlotFill> slotFillsForPivot = fillSlots(pivot, Pair.makePair(groups, sentences), false);
      for (KBPSlotFill fill : slotFillsForPivot) {
        graph.add(fill);
      }
      endTrack("Augmenting " + pivot);
    }

    // Return
    return graph;
  }

  @SuppressWarnings("UnusedDeclaration")
  protected void printGraph( String name, EntityGraph relationGraph) {
    VisualizationUtils.logGraph(name,
        CollectionUtils.map(relationGraph, in -> {
          String e1 = in.first.name;
          String e2 = in.second.name;
          List<String> relations = new ArrayList<>();
          for (KBPSlotFill fill : in.third) {
            relations.add(String.format("%s:%.2f", fill.key.relationName, fill.score.getOrElse(-1.0)));
          }
          return Triple.makeTriple(e1, e2, relations);
        }));
  }
}

