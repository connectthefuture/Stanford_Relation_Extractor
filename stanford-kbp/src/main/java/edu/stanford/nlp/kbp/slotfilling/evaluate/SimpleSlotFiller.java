package edu.stanford.nlp.kbp.slotfilling.evaluate;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import java.io.FileNotFoundException;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;
import java.util.function.Function;

import edu.stanford.nlp.kbp.entitylinking.AcronymMatcher;
import edu.stanford.nlp.kbp.slotfilling.classify.HeuristicRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.ModelType;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.StandardIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.process.*;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess.AnnotateMode;
import edu.stanford.nlp.kbp.slotfilling.process.RelationFilter.RelationFilterBuilder;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * An implementation of a SlotFiller.
 *
 * This class will fill slots given a query entity (complete with provenance).
 * The class depends on an IR component, the components in process to annotate documents, and the classifiers.
 *
 * @author Gabor Angeli
 */
public class SimpleSlotFiller implements SlotFiller {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Infer");

  // Defining Instance Variables
  protected final Properties props;
  public final KBPIR irComponent;
  public final KBPProcess process;
  public final RelationClassifier classifyComponent;
  public final Maybe<RelationFilter> relationFilterForFeaturizer;
  /** Additional classifiers to use -- e.g., for rule-based additions */
  public final RelationClassifier[] additionalClassifiers;
  
  /**
   * Used to keep track of all the (entity, slot fill candidate) pairs recovered via IR
   */
  protected final GoldResponseSet goldResponses;

  public SimpleSlotFiller(Properties props,
                      KBPIR ir,
                      KBPProcess process,
                      final RelationClassifier classify,
                      GoldResponseSet goldResponses
                      ) {
    this.props = props;
    this.process = process;
    this.classifyComponent = classify;
    
    if(Props.TEST_RELATIONFILTER_DO) {
      RelationFilterBuilder rfBuilder = new RelationFilterBuilder(in -> classify.classifyRelationsNoProvenance(in.first, in.second));
      //noinspection unchecked
      for(Class<RelationFilter.FilterComponent> filterComponent : Props.TEST_RELATIONFILTER_COMPONENTS) {
        rfBuilder.addFilterComponent(filterComponent);
      }
      this.relationFilterForFeaturizer = Maybe.Just(rfBuilder.make());
    }
    else {
      this.relationFilterForFeaturizer = Maybe.Nothing();
    }

    this.goldResponses = goldResponses;

    if (Props.TEST_GOLDIR) {
      this.irComponent = new StandardIR(props, goldResponses); // override the passed IR component
    } else {
      this.irComponent = ir;
    }

    // Construct the additional classifiers
    this.additionalClassifiers = new RelationClassifier[Props.TEST_AUXMODELS.length];
    for (int i = 0; i < Props.TEST_AUXMODELS.length; ++i) {
      this.additionalClassifiers[i] = Props.TEST_AUXMODELS[i].construct(props);
    }
  }

  @Override
  public List<KBPSlotFill> fillSlots(final KBPOfficialEntity queryEntity) {
    startTrack(BLUE, BOLD, "Annotating " + queryEntity);
    // IR + process
    Pair<? extends List<SentenceGroup>, ? extends Map<KBPair, CoreMap[]>> processedIR =
        Props.TEST_GOLDSLOTS ? Pair.makePair(new ArrayList<SentenceGroup>(), new HashMap<KBPair, CoreMap[]>())
            : queryAndProcessSentences(queryEntity, Props.TEST_SENTENCES_PER_ENTITY);
    // Slot fill
    List<KBPSlotFill> rtn =  fillSlots(queryEntity, processedIR, true);
    endTrack("Annotating " + queryEntity);
    return rtn;
  }

  protected List<KBPSlotFill> fillSlots(final KBPEntity queryEntity,
                                        final Pair<? extends List<SentenceGroup>, ? extends Map<KBPair, CoreMap[]>> datumsAndSentences,
                                        boolean doConsistency) {
    // Register datums we've seen
    final List<String> allSlotFills = new LinkedList<>();
    int totalDatums = 0;
    for (SentenceGroup tuple : datumsAndSentences.first) { 
      allSlotFills.add(tuple.key.slotValue); totalDatums += tuple.size(); 
    }
    if (datumsAndSentences.first.size() > 0 && doConsistency) { logger.log("Found " + datumsAndSentences.first.size() + " sentence groups (ave. " + (((double) totalDatums) / ((double) datumsAndSentences.first.size())) + " sentences per group)"); }

    startTrack("P(r | e_1, e_2)");
    startTrack("Classifying Relations");
    List<Counter<KBPSlotFill>> tuplesWithRelation;
    if (Props.TEST_GOLDSLOTS) {
      // Case: simply output the right answer every time. This can be useful to test consistency, and the scoring script
      tuplesWithRelation = new ArrayList<>();
      for (final KBPSlotFill response : goldResponses.correctFills()) {
        if (response.key.getEntity().equals(queryEntity)) {
          tuplesWithRelation.add(new ClassicCounter<KBPSlotFill>() {{
            setCount(response, response.score.orCrash());
          }});
        }
      }
    } else {
      tuplesWithRelation = CollectionUtils.map(datumsAndSentences.first, input -> {

        // vvv RUN CLASSIFIER vvv
        Counter<Pair<String, Maybe<KBPRelationProvenance>>> relationsAsStrings = classifyComponent.classifyRelations(input, Maybe.fromNull(datumsAndSentences.second.get(input.key)));
        // ^^^                ^^^

        // Convert to Probabilities
        Counter<KBPSlotFill> countsForKBPair = new ClassicCounter<>();
        for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : relationsAsStrings.entrySet()) {
          RelationType rel = RelationType.fromString(entry.getKey().first).orCrash();
          Maybe<KBPRelationProvenance> provenance = entry.getKey().second;
          double prob = entry.getValue();
          if (Props.TEST_PROBABILITYPRIORS) {
            prob = new Probabilities(irComponent, allSlotFills, entry.getValue())
                .ofSlotValueGivenRelationAndEntity(input.key.slotValue, rel, queryEntity);
          }
          KBPair key = input.key;
          if (!key.slotType.isDefined()) { logger.warn("slot type is not defined for KBPair: " + key); } // needed for some of the consistency checks
          countsForKBPair.setCount(KBPNew.from(key).rel(rel).provenance(provenance).score(prob).KBPSlotFill(), prob);
        }

        // output
        return countsForKBPair;
      });
    }
    endTrack("Classifying Relations");
    // Display predictions
    startTrack("Relation Predictions");
    java.text.DecimalFormat df = new java.text.DecimalFormat("0.000");
    for (Counter<KBPSlotFill> fillsByKBPair : tuplesWithRelation) {
      if (fillsByKBPair.size() > 0) {
        List<KBPSlotFill> bestRels = Counters.toSortedList(fillsByKBPair);
        StringBuilder b = new StringBuilder();
        b.append(fillsByKBPair.keySet().iterator().next().key.entityName).append(" | ");
        for (int i = 0; i < Math.min(bestRels.size(), 3); ++i) {
          b.append(bestRels.get(i).key.relationName)
              .append(" [").append(df.format(fillsByKBPair.getCount(bestRels.get(i)))).append("] | ");
        }
        b.append(fillsByKBPair.keySet().iterator().next().key.slotValue);
        if (doConsistency) { logger.log(b); }
      }
    }
    endTrack("Relation Predictions");
    endTrack("P(r | e_1, e_2)");

    // -- Convert to KBP data structures
    // Flatten the list of tuples, as we only care about (e_1, r, e_2) and don't want to group by e_2.
    List<KBPSlotFill> relations = new ArrayList<>();
    for (Counter<KBPSlotFill> counter : tuplesWithRelation) {
      for (Map.Entry<KBPSlotFill, Double> entry : counter.entrySet()) {
        if (Props.TEST_RELATIONEXTRACTOR_ALTERNATENAMES_DO || !entry.getKey().key.hasKBPRelation() || !entry.getKey().key.kbpRelation().isAlternateName()) {
          relations.add(entry.getKey());  // potentially filter out alternate names from relation extractor
        }
      }
    }

    // -- Add in rule-based extractions.
    //    note[gabor; 2013] these were very slow as of this note, and thus are only performed on the query entity.
    startTrack("P(e_2 | e_1, r)");
    if (queryEntity instanceof KBPOfficialEntity) {
      startTrack("Rule based additions");
      List<CoreMap> sentences = new ArrayList<>();
      for (Map.Entry<KBPair, CoreMap[]> entry : datumsAndSentences.second.entrySet()) {
        // Get sentences in official index (for alternate names only)
        for (CoreMap sent : entry.getValue()) {
          if (KBPRelationProvenance.isOfficialIndex(sent.get(KBPAnnotations.SourceIndexAnnotation.class))) { sentences.add(sent); }
        }

        // Run rule/pattern extractors
        SentenceGroup dummyDatum = SentenceGroup.empty(entry.getKey());
        for (RelationClassifier auxClassifier : this.additionalClassifiers) {
          Counter<Pair<String, Maybe<KBPRelationProvenance>>> auxRelations = auxClassifier.classifyRelations(dummyDatum, Maybe.Just(entry.getValue()));
          for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> fill : auxRelations.entrySet()) {
            if (doConsistency) { logger.log(entry.getKey().entityName + " | " + fill.getKey().first + " | " + entry.getKey().slotValue); }
            relations.add( KBPNew.from(entry.getKey()).rel(fill.getKey().first).provenance(fill.getKey().second).score(fill.getValue()).KBPSlotFill());
          }
        }
      }

      // Run alternate name extractor
      if (Props.TEST_RULES_ALTERNATENAMES_DO) {
        for (KBPSlotFill altName : AlternateNamesExtractor.extractSlots(queryEntity, sentences)) {
          if (doConsistency) { logger.log(altName.key.entityName + " | " + altName.key.relationName + " | " + altName.key.slotValue); }
          relations.add(altName);
        }
      }
      endTrack("Rule based additions");
    }
    if (doConsistency) { logger.log("" + relations.size() + " slots extracted"); }
    if (queryEntity instanceof KBPOfficialEntity) { for (KBPSlotFill slot : relations) { goldResponses.registerResponse(slot); } }
    endTrack("P(e_2 | e_1, r)");

    // -- Early exit (without consistency)
    if (!doConsistency) { return relations; }

    // -- Run consistency checks
    startTrack("Consistency and Inference");
    // Run consistency pass 1 (after de-duplicating)
    List<KBPSlotFill> cleanRelations
        = Props.TEST_CONSISTENCY_DO ? SlotfillPostProcessor.unary(irComponent).postProcess(queryEntity, relations, goldResponses) : relations;
    logger.log("" + cleanRelations.size() + " slot fills remain after consistency (pass 1)");
    // Filter on missing provenance
    List<KBPSlotFill> withProvenance = new ArrayList<>(cleanRelations.size());
    for (KBPSlotFill fill : cleanRelations) {
      KBPSlotFill augmented = KBPNew.from(fill).provenance(findBestProvenance(queryEntity, fill)).KBPSlotFill();
      if (!Props.TEST_PROVENANCE_DO || (augmented.provenance.isDefined() && augmented.provenance.get().isOfficial())) {
        withProvenance.add(augmented);
      } else {
        goldResponses.discardNoProvenance(fill);
      }
    }
    for (KBPSlotFill slot : withProvenance) { goldResponses.registerResponse(slot); } // re-register after provenance
    logger.log("" + withProvenance.size() + " slot fills remain after provenance");
    // Run consistency pass 2
    List<KBPSlotFill> consistentRelations = finalConsistencyAndProvenancePass(queryEntity, new ArrayList<KBPSlotFill>(new HashSet<KBPSlotFill>(withProvenance)), goldResponses);
    endTrack("Consistency and Inference");
   
    // -- Print Judgements
    if (queryEntity instanceof KBPOfficialEntity) {
      prettyLog(goldResponses.loggableForEntity((KBPOfficialEntity) queryEntity, Maybe.Just(irComponent)));
    }

    return consistentRelations;
  }

  //
  // Public Utilities
  //

  /**
   * Runs a final pass for consistency and get any provenances that haven't been retrieved yet.
   *
   * @param queryEntity The entity we we are checking slots for
   * @param slotFills The candidate slot fills to filter for consistency
   * @param responseChecklist The response checklist to register fills that have been added or removed
   * @return A list of slot fills, guaranteed to be consistent and with provenance
   */
  protected List<KBPSlotFill> finalConsistencyAndProvenancePass(KBPEntity queryEntity, List<KBPSlotFill> slotFills, GoldResponseSet responseChecklist) {
    // Run consistency pass 2
    List<KBPSlotFill> consistentRelations
      = Props.TEST_CONSISTENCY_DO ? SlotfillPostProcessor.global(irComponent).postProcess(queryEntity, slotFills, responseChecklist) : slotFills;
    logger.log("" + consistentRelations.size() + " slot fills remain after consistency (pass 2)");
    // Run provenance pass 2
    List<KBPSlotFill> finalRelations = new ArrayList<>();
    // (last pass to make sure we have provenance)
    for (KBPSlotFill fill : consistentRelations) {
      assert fill != null;
      KBPSlotFill augmented = KBPNew.from(fill).provenance(findBestProvenance(queryEntity, fill)).KBPSlotFill();
      if (!Props.TEST_PROVENANCE_DO || (augmented.provenance.isDefined() && augmented.provenance.get().isOfficial())) {
        finalRelations.add(augmented);
      } else {
        responseChecklist.discardNoProvenance(fill);
      }
    }
    logger.log("" + consistentRelations.size() + " slot fills remain after final provenance check");

    if (!Props.TEST_GOLDSLOTS && Props.TRAIN_MODEL != ModelType.GOLD && !Props.TEST_GOLDIR && Props.TEST_PROVENANCE_DO) {
      for (KBPSlotFill fill : finalRelations) { if (!fill.provenance.isDefined() || !fill.provenance.get().isOfficial()) { throw new IllegalStateException("Invalid provenance for " + fill); } }
    }
    return finalRelations;
  }

  /**
   * Query and annotate a KBPOfficialEntity to get a featurized and annotated KBPTuple.
   *
   * @param entity The entity to query. This is a fancy way of saying "Obama"
   * @return A list of featurized datums, and a collection of raw sentences (for rule-based annotators)
   */
  @SuppressWarnings("unchecked")
  private Pair<List<SentenceGroup>, Map<KBPair, CoreMap[]>> queryAndProcessSentences(KBPOfficialEntity entity, int sentencesPerEntity) {
    startTrack("Processing " + entity + " [" + sentencesPerEntity + " sentences max]");

    // -- IR
    // Get supporting sentences
    List<CoreMap> rawSentences;
    try {
      rawSentences = irComponent.querySentences(entity,
          entity.representativeDocumentId().isDefined() ? new HashSet<>(Arrays.asList(entity.queryId.get())) : new HashSet<String>(),
          sentencesPerEntity);
    } catch (Exception e) {
      e.printStackTrace();
      logger.err(RED, "Querying failed! Is Lucene set up at the paths:  " + Arrays.toString(Props.INDEX_PATHS) + "?");
      rawSentences = Collections.EMPTY_LIST;
    }
    // Get datums from sentences.
    Redwood.startTrack("Annotating " + rawSentences.size() + " sentences...");
    List<CoreMap> supportingSentences = process.annotateSentenceFeatures(entity, rawSentences, AnnotateMode.ALL_PAIRS);
    Redwood.endTrack("Annotating " + rawSentences.size() + " sentences...");

    // -- Process
    Annotation annotation = new Annotation("");
    annotation.set(CoreAnnotations.SentencesAnnotation.class, supportingSentences);
    Redwood.forceTrack("Featurizing " + annotation.get(CoreAnnotations.SentencesAnnotation.class).size() + " sentences...");
    Map<KBPair, Pair<SentenceGroup, List<CoreMap>>> datums = process.featurizeWithSentences(annotation, relationFilterForFeaturizer);
    Redwood.endTrack("Featurizing " + annotation.get(CoreAnnotations.SentencesAnnotation.class).size() + " sentences...");
    // Register this as a datum we've seen
    logger.log("registering slot fills [" + datums.size() + " KBPairs]...");
    endTrack("Processing " + entity + " [" + sentencesPerEntity + " sentences max]");

    // -- Return
    List<SentenceGroup> groups = new ArrayList<>();
    Map<KBPair, CoreMap[]> sentences = new HashMap<>();
    for (Map.Entry<KBPair, Pair<SentenceGroup, List<CoreMap>>> datum : datums.entrySet()) {
      if (datum.getKey().getEntity().equals(entity)) {
        groups.add( Props.HACKS_DISALLOW_DUPLICATE_DATUMS ? datum.getValue().first.removeDuplicateDatums() : datum.getValue().first );
        sentences.put(datum.getKey(), datum.getValue().second.toArray(new CoreMap[datum.getValue().second.size()]));
      }
    }
    return Pair.makePair(groups, sentences);
  }



  protected Maybe<KBPRelationProvenance> findBestProvenance(final KBPEntity entity, final KBPSlotFill fill) {
    if (!Props.TEST_PROVENANCE_DO) { return fill.provenance.orElse(Maybe.Just(new KBPRelationProvenance("unk_id", "/unk/index"))); }
    if(fill.provenance.isDefined() && fill.provenance.get().isOfficial()) {
      return fill.provenance;
    }

    //check if the slot fill is set up rule based classifiers. we will try to get provenance for slot values set by rules because they are presumably high precision
    boolean setByRules = false;
    if(fill.provenance.isDefined() && fill.provenance.get().getClassifierClass().isDefined() && (fill.provenance.get().getClassifierClass().get() == HeuristicRelationExtractor.class)){
      setByRules = true;
    }

    if(setByRules){
      if(!Props.TEST_PROVENANCE_RECOVER_RULES)
        return Maybe.Nothing();
    }else if (!Props.TEST_PROVENANCE_RECOVER) {
      return Maybe.Nothing();
    }

    startTrack("Provenance For " + fill);
    final Pointer<KBPRelationProvenance> bestProvenance = new Pointer<>();
    double bestProvenanceProbability = -0.01;
    // Try the cache
    if (Props.CACHE_PROVENANCE_DO && !Props.CACHE_PROVENANCE_REDO) {
      PostgresUtils.withKeyProvenanceTable(Props.DB_TABLE_PROVENANCE_CACHE, new PostgresUtils.KeyProvenanceCallback() {
        @Override
        public void apply(Connection psql) throws SQLException {
          for (KBPRelationProvenance prov : get(psql, Props.DB_TABLE_PROVENANCE_CACHE, keyToString(fill.key))) {
            if (!prov.sentenceIndex.isDefined()) {
              warn("retrieved provenance that didn't have a sentence index -- re-computing");
            } else if (!prov.isOfficial()) {
              warn("retrieved unofficial provenance -- recomputing");
            } else {
              bestProvenance.set(prov);
            }
          }
        }
      });
    }
    if (!bestProvenance.dereference().isDefined()) {
      logger.debug("provenance cache miss!");

      // Try to use provenance from original classifier
      if (!bestProvenance.dereference().isDefined() && fill.provenance.isDefined() && fill.provenance.get().sentenceIndex.isDefined() &&
          fill.provenance.get().isOfficial()) {
        logger.debug("using provenance from classifier");
        bestProvenance.set(fill.provenance.get());
        if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
      }

      // Query
      if (!bestProvenance.dereference().isDefined()) {
        KBTriple key = fill.key;
        // Get String forms of entity and slot value
        String entityName = entity.name;
        if (fill.provenance.isDefined() && fill.provenance.get().entityMentionInSentence.isDefined() && fill.provenance.get().containingSentenceLossy.isDefined()) {
          entityName = CoreMapUtils.sentenceSpanString(fill.provenance.get().containingSentenceLossy.get(), fill.provenance.get().entityMentionInSentence.get());
        }
        String slotValue = key.slotValue;
        if (fill.provenance.isDefined() && fill.provenance.get().slotValueMentionInSentence.isDefined() && fill.provenance.get().containingSentenceLossy.isDefined()) {
          slotValue = CoreMapUtils.sentenceSpanString(fill.provenance.get().containingSentenceLossy.get(), fill.provenance.get().slotValueMentionInSentence.get());
        }
        List<CoreMap> potentialProvenances = this.irComponent.querySentences(key.getEntity(), key.getSlotEntity(), 25, true);
        if (!key.slotValue.equals(slotValue)) { potentialProvenances.addAll(this.irComponent.querySentences(key.getEntity(), key.getSlotEntity(), 25, true)); }
        if (!entity.name.equals(entityName)) { potentialProvenances.addAll(this.irComponent.querySentences(key.getEntity(), key.getSlotEntity(), 25, true)); }
        for(CoreMap sentence : potentialProvenances) {
          if (sentence.get(CoreAnnotations.TokensAnnotation.class).size() > 150) { continue; }
          // Error check
          if (!sentence.containsKey(KBPAnnotations.SourceIndexAnnotation.class) && sentence.get(KBPAnnotations.SourceIndexAnnotation.class).toLowerCase().endsWith(Props.INDEX_OFFICIAL.getName().toLowerCase())) {
            warn("Queried a document which is purportedly not from the official index!");
            continue;
          }


          // Try to classify provenance
          Annotation ann = new Annotation("");
          List<CoreMap> sentences = Arrays.asList(sentence);
          sentences = this.process.annotateSentenceFeatures(key.getEntity(), sentences);
          ann.set(CoreAnnotations.SentencesAnnotation.class, sentences);
          Map<KBPair, SentenceGroup> datums = this.process.featurize(ann);
          // Get the best key to match to
          KBPair pair = KBPNew.from(key).KBPair();  // default
          if (!datums.containsKey(pair)) {  // try to find a close match
            for (KBPair candidate : datums.keySet()) {
              if ((candidate.getEntity().equals(key.getEntity()) || candidate.getEntity().name.equals(entityName)) &&  // the entities must match
                 (candidate.slotValue.toLowerCase().contains(slotValue.toLowerCase()) ||  // try with the inferred "original" slot value
                     slotValue.toLowerCase().contains(candidate.slotValue.toLowerCase()) ||
                     AcronymMatcher.isAcronym(candidate.slotValue, slotValue.split("\\s+")) ||
                     AcronymMatcher.isAcronym(slotValue, candidate.slotValue.split("\\s+")) ||
                     candidate.slotValue.toLowerCase().contains(key.slotValue.toLowerCase()) ||  // try with the defined slot value
                     key.slotValue.toLowerCase().contains(candidate.slotValue.toLowerCase()) ||
                     AcronymMatcher.isAcronym(candidate.slotValue, key.slotValue.split("\\s+")) ||
                     AcronymMatcher.isAcronym(key.slotValue, candidate.slotValue.split("\\s+"))  )  ) {
                logger.debug("using key: " + candidate);
                pair = candidate;
                break;
              }
            }
          }

          // Classify
          if (datums.containsKey(pair)) {
            Pair<Double, Maybe<KBPRelationProvenance>> candidate = this.classifyComponent.classifyRelation(datums.get(pair), RelationType.fromString(key.relationName).orCrash(), Maybe.<CoreMap[]>Nothing());
            if (!bestProvenance.dereference().isDefined() || candidate.first > bestProvenanceProbability) {
              // Candidate provenance found
              boolean updated = false;
              if (candidate.second.isDefined() && candidate.second.get().sentenceIndex.isDefined() && candidate.second.get().isOfficial()) {
                // Case: can use the provenance from classification (preferred)
                bestProvenance.set(candidate.second.get());
                updated = true;
                if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
              } else {
                // Case: can compute a provenance from the sentence
                for (KBPRelationProvenance provenance : KBPRelationProvenance.compute(sentence, fill.key)) { bestProvenance.set(provenance); updated = true; }
                if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
              }
              // Perform some updating if a provenance was set
              if (updated) {
                if (candidate.first > bestProvenanceProbability) { bestProvenanceProbability = candidate.first; }
                logger.debug("using: " + CoreMapUtils.sentenceToMinimalString(sentence));
              }
            }
            /*if (!bestProvenance.dereference().isDefined()){
              //Use additional classifiers if available, like rule based
              for (RelationClassifier auxClassifier : this.additionalClassifiers) {
                Pair<Double, Maybe<KBPRelationProvenance>> candidateRules = auxClassifier.classifyRelation(datums.get(pair), RelationType.fromString(key.relationName).orCrash(), Maybe.<CoreMap[]>Nothing());
                if (!bestProvenance.dereference().isDefined() || candidateRules.first > bestProvenanceProbability) {
                  // Candidate provenance found
                  boolean updated = false;
                  if (candidateRules.second.isDefined() && candidateRules.second.get().sentenceIndex.isDefined() && candidateRules.second.get().isOfficial()) {
                    // Case: can use the provenance from classification (preferred)
                    bestProvenance.set(candidateRules.second.get());
                    updated = true;
                    if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
                  } else {
                    // Case: can compute a provenance from the sentence
                    for (KBPRelationProvenance provenance : KBPRelationProvenance.compute(sentence, fill.key)) { bestProvenance.set(provenance); updated = true; }
                    if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
                  }
                  // Perform some updating if a provenance was set
                  if (updated) {
                    if (candidateRules.first > bestProvenanceProbability) { bestProvenanceProbability = candidateRules.first; }
                    logger.debug("using: " + CoreMapUtils.sentenceToMinimalString(sentence));
                  }
                }
              }
            }*/
          }
        }

        // Backup if classification fails (pick shortest IR result)
        // TODO(gabor) this is awful correctness-wise. We should look into figuring out if we can fix this
        if(Props.TEST_PROVENANCE_RECOVER_NONCLASSIFY){
          logger.log("Recovering provenance finding using the shortest sentence in IR");
          if (!bestProvenance.dereference().isDefined()) {
            int minLength = Integer.MAX_VALUE;
            CoreMap shortestProvenance = null;
            for(CoreMap sentence : potentialProvenances) {
              if (sentence.get(CoreAnnotations.TokensAnnotation.class).size() < minLength) {
                minLength = sentence.get(CoreAnnotations.TokensAnnotation.class).size();
                shortestProvenance = sentence;
              }
            }
            if (minLength < 50 && shortestProvenance != null) {
              logger.warn("using first IR result: " + CoreMapUtils.sentenceToMinimalString(shortestProvenance));
              for (KBPRelationProvenance provenance : KBPRelationProvenance.compute(shortestProvenance, fill.key)) { bestProvenance.set(provenance); }
              if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }

            }
          }
        }
      }

      // Do Cache
      if (Props.CACHE_PROVENANCE_DO && bestProvenance.dereference().isDefined()) {
        PostgresUtils.withKeyProvenanceTable(Props.DB_TABLE_PROVENANCE_CACHE, new PostgresUtils.KeyProvenanceCallback() {
          @Override
          public void apply(Connection psql) throws SQLException {
            put(psql, Props.DB_TABLE_PROVENANCE_CACHE, keyToString(fill.key), bestProvenance.dereference().orCrash());
            if (Props.KBP_EVALUATE && !psql.getAutoCommit()) { psql.commit(); }  // slower, but allows for stopping a run halfway through
          }
        });
      }
    }

    // Return
    if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
    debug(bestProvenance.dereference().isDefined()
            ? "found provenance" + (bestProvenance.dereference().get().isOfficial() ? " (is official) " : " ")
            : "no provenance!");
    endTrack("Provenance For " + fill);
    return bestProvenance.dereference();
  }

}


