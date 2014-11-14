package edu.stanford.nlp.kbp.slotfilling.train;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import java.io.*;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;
import java.util.function.Function;

import au.com.bytecode.opencsv.CSVReader;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.classify.KBPDataset;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.classify.TrainingStatistics;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.process.FeatureFactory;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * Finds supporting relational data and trains a classifiers.
 */
public class KBPTrainer {
  public static enum UnlabeledSelectMode { NEGATIVE, NOT_POSITIVE, NOT_LABELED, NOT_POSITIVE_WITH_NEGPOS, NOT_LABELED_WITH_NEGPOS }
  public static enum MinimizerType { QN, SGD, SGDTOQN }

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Train");

  protected KBPIR querier;
  protected KBPProcess process;
  protected RelationClassifier classifier;

  /**
   * A map from sentence gloss key to annotated label, as collected from active learning
   */
  private final Map<String, String> annotationForSentence = new LinkedHashMap<>();

  /**
   * Create a new KBPTrainer -- the entry point for training a classifier.
   * @param querier The IR component to collect data from.
   * @param process The process component to featurize data with.
   * @param classifier The classifier to train with the collected data.
   */
  public KBPTrainer(KBPIR querier, KBPProcess process, RelationClassifier classifier) {
    // Set instance variables
    this.querier = querier;
    this.process = process;
    this.classifier = classifier;

    // Read annotated labels
    if (Props.TRAIN_ANNOTATED_SENTENCES_DO) {
      // (read keys, if applicable)
      Maybe<? extends Set<String>> validKeys = Maybe.Nothing();
      if (Props.TRAIN_ANNOTATED_SENTENCES_KEYS != null && !Props.TRAIN_ANNOTATED_SENTENCES_KEYS.getPath().trim().equals("") && !Props.TRAIN_ANNOTATED_SENTENCES_KEYS.getPath().equals("/dev/null")) {
        try {
          validKeys = Maybe.Just(new HashSet<>(Arrays.asList(IOUtils.slurpFile(Props.TRAIN_ANNOTATED_SENTENCES_KEYS).split("\n"))));
        } catch (IOException e) {
          logger.err("could not read annotated sentence keys file at " + Props.TRAIN_ANNOTATED_SENTENCES_KEYS + "; disallowing any annotated sentences");
          validKeys = Maybe.Just(new HashSet<String>());
        }
      }
      // (read data)
      for (File file : Props.TRAIN_ANNOTATED_SENTENCES_DATA) {
        try {
          CSVReader reader = new CSVReader(new FileReader(file));
          @SuppressWarnings("UnusedAssignment") // ignore the header
          String[] nextLine = reader.readNext();
          while ( (nextLine = reader.readNext()) != null ) {
            String key = nextLine[0];
            if (!validKeys.isDefined() || validKeys.get().contains(key)) {
              annotationForSentence.put(key, "no_relation".equals(nextLine[1]) ? RelationMention.UNRELATED : RelationType.fromString(nextLine[1]).orCrash("Unknown annotated relation type: " + nextLine[1]).canonicalName);
            }
          }
        } catch (IOException e) {
          logger.err("Could not read annotation file: " + file.getPath());
        }
      }
      logger.log("read " + annotationForSentence.size() + " labelled sentence annotations");
    }
  }

  /**
   * Train the classifier on the data provided
   * @param dataset - List of training data
   * @return - classifier
   */
  public Pair<RelationClassifier, TrainingStatistics> trainOnData( KBPDataset<String, String> dataset ) {
    TrainingStatistics statistics = classifier.train(dataset);
    return Pair.makePair(classifier, statistics);
  }

  /**
   * Find relevant sentences, construct relation datums and train the classifier
   * @param tuples - List of true (entity, rel, entity) triples.
   * @return - classifier
   */
  public Pair<RelationClassifier, TrainingStatistics> trainOnTuples( List<KBPair> tuples ) {
    return trainOnData(makeDataset(findDatumsFromSeedQueries(tuples)));
  }

  /**
   * Creates a (lazy) iterator of datums given a collection of query tuples.
   * If a datum is found in the cache, it is returned. Otherwise, the datum is created lazily.
   * NOTE: When fetching from cached, the datums can include those not in the requested tuples
   *       and the ordering is different from the input tuples
   * @param tuples The tuples to query
   * @return A lazy iterator of {@link SentenceGroup}s corresponding to the datums for that query.
   *         Note that this includes both positive and negative datums.
   */
  public Iterator<SentenceGroup> findDatumsFromSeedQueries(final Collection<? extends KBPair> tuples) {
    // Shortcut if we're reading only cached data
    // This has the advantage of doing a linear scan, rather than n * O( log(n) ) random disk accesses,
    // however, it also ignores the passed tuples and returns every datum in the cache.
    if (Props.CACHE_DATUMS_SLURP) {
      final Pointer<Iterator<SentenceGroup>> allDatums = new Pointer<>();
      final Set<KBPair> tupleSet = new HashSet<>(tuples);
      PostgresUtils.withKeyDatumTable(Props.DB_TABLE_DATUM_CACHE, new PostgresUtils.KeyDatumCallback() {
        @Override
        public void apply(Connection psql) throws SQLException {
          logger.log("connected to datum table");
          allDatums.set(CollectionUtils.filter(this.values(psql, Props.DB_TABLE_DATUM_CACHE + (Props.CACHE_DATUMS_ORDER ? "_asc" : ""), 1000), in -> !in.isEmpty() && (tupleSet.contains(KBPNew.from(in.key).KBPair()) || !querier.getKnowledgeBase().contains(in.key))));
          logger.log("created datum iterator");
        }
      });
      if (allDatums.dereference().isDefined()) { return allDatums.dereference().get(); }
    }

    // Else, start caching!
    final boolean doCache = Props.CACHE_DATUMS_DO;
    final boolean redoCache = Props.CACHE_DATUMS_REDO;
    final boolean doSentenceCache;
    synchronized (Props.PROPERTY_CHANGE_LOCK) { doSentenceCache = Props.CACHE_SENTENCES_DO; }
    return CollectionUtils.iteratorFromMaybeIterableFactory(new Factory<Maybe<Iterable<SentenceGroup>>>() {
      /** The tuples to iterate over */
      Iterator<? extends KBPair> iter = tuples.iterator();
      /**
       * Pedantic detail: we need to make sure that we don't return the same mention (datum in a {@link SentenceGroup})
       * multiple times.
       * Optimally, we would collect all the {@link SentenceGroup}s and merge the relevant datums; this is, effectively,
       * what happens in the datum cache -- and we get this for free if we ignore uncached datums and read from the cache
       * directly. However, when reading datums from here we have to load them lazily (or face memory explosion), and don't have
       * the benefit of foresight, so we have to simply disallow future sentence groups that share a key with something already
       * returned.
       *
       */
      Set<KBPair> keysToNotDuplicate = new HashSet<>();

      @Override
      public Maybe<Iterable<SentenceGroup>> create() {
        if (iter.hasNext()) {
          final KBPair key = iter.next();
          final Pointer<Set<SentenceGroup>> datums = new Pointer<>();

          // Try Cache
          if (doCache && !redoCache) {
            PostgresUtils.withKeyDatumTable(Props.DB_TABLE_DATUM_CACHE, new PostgresUtils.KeyDatumCallback() {
              @Override
              public void apply(Connection psql) throws SQLException {
                final Maybe<SentenceGroup> cachedValue = get(psql, Props.DB_TABLE_DATUM_CACHE, keyToString(key));
                if (cachedValue.isDefined()) {
                  datums.set(new HashSet<SentenceGroup>() {{
                    add(cachedValue.get());
                  }});
                }
              }
            });
          }

          // Run Featurizer, if cache missed
          if (!datums.dereference().isDefined()) {
            startTrack(key.toString());
            KBPEntity entity1 = key.getEntity();
            String entity2 = key.slotValue;

            // ----- REAL WORK DONE HERE -----
            // vv (1) Query Sentence In Lucene vv
            // Query just for entity1 and entity2 without reln (
            // so we don't bias the training data with what we think is indicative of the relation)
            List<CoreMap> sentences;
            synchronized (Props.PROPERTY_CHANGE_LOCK) {
              boolean saveDoSentenceCache = Props.CACHE_SENTENCES_DO;
              Props.CACHE_SENTENCES_DO = doSentenceCache;
              sentences = querier.querySentences(entity1.name, entity2,
                  (key instanceof KBTriple ? Maybe.Just(((KBTriple) key).relationName) : Maybe.<String>Nothing()),
                  Props.TRAIN_SENTENCES_PER_ENTITY);
              Props.CACHE_SENTENCES_DO = saveDoSentenceCache;
            }
            // ^^
            logger.logf("Found %d sentences for %s", sentences.size(), key);
            // vv (2) Annotate Sentence vv
            sentences = process.annotateSentenceFeatures(entity1, sentences);
            // ^^
            logger.logf("Keeping %d sentences after annotation", sentences.size());
            HashMap<KBPair, SentenceGroup> featurized;
            if (sentences.size() > 0) {
              try {
                // Get datums from sentences.
                Annotation annotation = new Annotation("");
                annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
                // vv (3) Featurize Sentence vv
                featurized = process.featurize(annotation);
                // ^^
                datums.set(new HashSet<>(featurized.values()));
              } catch (RuntimeException e) {
                logger.warn(e);
              }
            }
            // ----- DONE WITH REAL WORK -----

            // Cache
            if (doCache) {
              PostgresUtils.withKeyDatumTable(Props.DB_TABLE_DATUM_CACHE, new PostgresUtils.KeyDatumCallback() {
                @Override
                public void apply(Connection psql) throws SQLException {
                  int numCached = 0;
                  if (datums.dereference().isDefined()) {
                    for (SentenceGroup group : datums.dereference().get()) {
                      append(psql, Props.DB_TABLE_DATUM_CACHE, keyToString(group.key), group);
                      numCached += 1;
                    }
                  } else {
                    append(psql, Props.DB_TABLE_DATUM_CACHE, keyToString(key), SentenceGroup.empty(key));
                  }
                  if (numCached > 0) {
                    logger.logf("cached %d non-empty sentence groups", numCached);
                  }
                }
              });
            }
            endTrack(key.toString());
          }

          // Return
          if (datums.dereference().isDefined()) {
            ArrayList<SentenceGroup> values = new ArrayList<>();
            for (SentenceGroup datum : datums.dereference().get()) {
              if (keysToNotDuplicate.add(datum.key)) {
                values.add(datum.removeDuplicateDatums());
              }
            }
            Collections.sort(values);
            return Maybe.Just((Iterable<SentenceGroup>) values);
          } else {
            return Maybe.Nothing();
          }
        } else {
          return null;
        }
      }
    });
  }

  /**
   * Get only the supervised data from the sentence gloss cache;
   * This will featurize the data on the fly.
   *
   * Note that this method is somewhat lossy, as the entity and slot value can't be recovered
   * exactly from the sentence gloss key alone.
   */
  public Iterator<SentenceGroup> supervisedData(int numDatums) {
    ArrayList<SentenceGroup> supervisedData = new ArrayList<>();
    OUTER: for (Map.Entry<String, String> datum : annotationForSentence.entrySet()) {
      if (supervisedData.size() == numDatums) { break; }
      for (CoreMap sentence : this.process.recoverSentenceGloss(datum.getKey())) {
        // Create the mock "sentences" array
        List<CoreMap> sentences = new ArrayList<>();
        sentences.add(sentence);
        // Create the target entity
        List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
        // (get the name)
        Span entitySpan = sentence.get(KBPAnnotations.EntitySpanAnnotation.class);
        String name = tokens.get(entitySpan.start()).get(CoreAnnotations.AntecedentAnnotation.class);
        if (name != null) {
          List<CoreLabel> entityTokens = tokens.subList(entitySpan.start(), entitySpan.end());
          List<String> entityGloss = new ArrayList<>();
          for (CoreLabel token : entityTokens) {
            entityGloss.add(token.word());
          }
          name = StringUtils.join(entityGloss, " ");
        }
        // (get the type)
        Maybe<NERTag> type = Maybe.Nothing();
        for (int i : entitySpan) {
          type = type.orElse(NERTag.fromString(tokens.get(i).ner()));
        }
        if (!type.isDefined() || !(type.get() == NERTag.PERSON || type.get() == NERTag.ORGANIZATION)) {
          if (datum.getValue().startsWith("per")) {
            type = Maybe.Just(NERTag.PERSON);
          } else if (datum.getValue().startsWith("org")) {
            type = Maybe.Just(NERTag.ORGANIZATION);
          } else {
            type = Maybe.Just(NERTag.PERSON);  // TODO(gabor) shouldn't have this backoff
          }
        }
        // (create the entity)
        KBPEntity entity = KBPNew.entName(name).entType(type.orCrash("Could not determine entity type for " + name)).KBPEntity();
        // Create the slot value
        Span slotSpan = sentence.get(KBPAnnotations.SlotValueSpanAnnotation.class);
        List<CoreLabel> slotTokens = tokens.subList(slotSpan.start(), slotSpan.end());
        List<String> slotGloss = new ArrayList<>();
        String slot = tokens.get(slotSpan.start()).get(CoreAnnotations.AntecedentAnnotation.class);
        if (slot != null) {
          for (CoreLabel token : slotTokens) {
            slotGloss.add(token.word());
          }
          slot = StringUtils.join(slotGloss, " ");
        }
        // Featurize
        sentences = process.annotateSentenceFeatures(entity, sentences);
        Annotation annotation = new Annotation("");
        annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
        HashMap<KBPair, SentenceGroup> featurized = process.featurize(annotation);
        // Add sentence group
        for (Map.Entry<KBPair, SentenceGroup> entry : featurized.entrySet()) {
          if (entry.getKey().entityName.equals(name) && entry.getKey().slotValue.equals(slot)) {
            logger.log("found datum: " + entry.getKey());
            supervisedData.add(entry.getValue());
            continue OUTER;
          }
        }
      }
      logger.warn("could not find datum for key: " + datum.getKey());
    }
    // Return
    return supervisedData.iterator();
  }

  /**
   * @see KBPTrainer#makeDataset(java.util.Iterator, edu.stanford.nlp.kbp.common.Maybe)
   * @param datums list of sentence examples.
   * @return A KBP Dataset
   */
  @SuppressWarnings("UnusedDeclaration")
  public KBPDataset<String, String> makeDataset(Map<KBPair,SentenceGroup> datums) {
    return makeDataset(datums.values().iterator(), Maybe.<Map<KBPair, Set<String>>>Nothing());
  }

  /**
   * @see KBPTrainer#makeDataset(java.util.Iterator, edu.stanford.nlp.kbp.common.Maybe)
   * @param datums list of sentence examples.
   * @param positiveRelations The known positive labels
   * @return A KBP Dataset
   */
  public KBPDataset<String, String> makeDataset(Map<KBPair,SentenceGroup> datums,
                                                Map<KBPair, Set<String>> positiveRelations) {
    return makeDataset(datums.values().iterator(), Maybe.Just(positiveRelations));
  }

  /**
   * @see KBPTrainer#makeDataset(java.util.Iterator, edu.stanford.nlp.kbp.common.Maybe)
   * @param datums list of sentence examples.
   * @return A KBP Dataset
   */
  public KBPDataset<String, String> makeDataset(Iterator<SentenceGroup> datums) {
    return makeDataset(datums, Maybe.<Map<KBPair, Set<String>>>Nothing());
  }

  /**
   * A helper function to compute positive and negative relations for a given datum.
   * @param datum The datum to compute relations for.
   * @param ir The IR component, for accessing the knowledge base.
   * @param annotationsForSentence The known annotations for this sentence, from active learning.
   * @return A pair of positive relations and negative relations for this datum.
   */
  public static Pair<Set<String>, Set<String>> computePositiveAndNegativeRelations(
      SentenceGroup datum, KBPIR ir, Function<String, String> annotationsForSentence,
      Function<KBPair, Set<String>> positiveRelationProvider) {
    Set<String> initialPositiveRelations = positiveRelationProvider.apply(datum.key);
    Set<String> cleanPositiveRelations = cleanPositiveRelations(initialPositiveRelations, datum, annotationsForSentence);
    Set<String> cleanNegativeRelations = computeNegativeRelations(cleanPositiveRelations, datum, ir);
    return Pair.makePair(cleanPositiveRelations, cleanNegativeRelations);
  }

  /**
   * A static helper for cleaning up the positive labels for a sentence
   * @param positiveLabels The initial positive labels from the Knowledge Base
   * @param group The sentence group we are extracting these relations for.
   * @param annotationForSentence A provider of annotated sentences.
   * @return The final set of positive relations for this example.
   */
  public static Set<String> cleanPositiveRelations(Set<String> positiveLabels,
                                                   SentenceGroup group,
                                                   Function<String, String> annotationForSentence) {
    KBPair key = group.key;
    // Get all positive labels
    if (positiveLabels == null) { positiveLabels = new HashSet<>(); }
    // Remove impossible positive labels (our KB is noisy...)
    if (Props.TRAIN_FIXKB) {
      Iterator<String> positiveLabelIter = positiveLabels.iterator();
      while (positiveLabelIter.hasNext()) {
        for (RelationType relation : RelationType.fromString(positiveLabelIter.next())) {
          if (relation.entityType != key.entityType || // if doesn't match entity type (e.g., a per: relation for an ORG)
              (key.slotType.isDefined() &&  // ... or doesn't match slot type (with fudge allowed for non-regexner tags, in case it hasn't been run)
                  key.slotType.get().isRegexNERType && !relation.validNamedEntityLabels.contains(key.slotType.get())) ) {
            positiveLabelIter.remove();  // remove it
          }
        }
      }
    }
    // Tweak employee_of and member_of (collapsed completely in 2013)
    if (positiveLabels.contains(RelationType.PER_MEMBER_OF.canonicalName)) {
      positiveLabels.remove(RelationType.PER_MEMBER_OF.canonicalName);
      positiveLabels.add(RelationType.PER_EMPLOYEE_OF.canonicalName);
    }
    // Augment positive labels with annotated examples
    @SuppressWarnings("unchecked")
    Maybe<String>[] annotatedLabels = new Maybe[group.size()];
    for (int sentenceI = 0; sentenceI < annotatedLabels.length; ++sentenceI) {
      // get annotated labels
      if (group.sentenceGlossKeys.isDefined()) {
        annotatedLabels[sentenceI] = Maybe.fromNull(annotationForSentence.apply(group.sentenceGlossKeys.get().get(sentenceI)));
      } else {
        annotatedLabels[sentenceI] = Maybe.Nothing();
      }
      // If we have annotated
      if (annotatedLabels[sentenceI].isDefined()) {
        Maybe<RelationType> rel = RelationType.fromString(annotatedLabels[sentenceI].get());
        if (rel.isDefined() && rel.get().entityType != key.entityType) {
          // ignore label
          annotatedLabels[sentenceI] = Maybe.Nothing();
        } else {
          // Clean up employee / member distinction (these are common mistakes, and in fact collapsed in 2013)
          if (positiveLabels.contains(RelationType.PER_MEMBER_OF.canonicalName) && annotatedLabels[sentenceI].get().equals(RelationType.PER_EMPLOYEE_OF.canonicalName)) {
            positiveLabels.remove(RelationType.PER_MEMBER_OF.canonicalName);
            positiveLabels.add(RelationType.PER_EMPLOYEE_OF.canonicalName);
            if (Utils.assertionsEnabled()) { throw new AssertionError("Should already have prohibited " + RelationType.PER_MEMBER_OF.canonicalName + " as a positive label!"); }
          } else if (positiveLabels.contains(RelationType.PER_EMPLOYEE_OF.canonicalName) && annotatedLabels[sentenceI].get().equals(RelationType.PER_MEMBER_OF.canonicalName)) {
            annotatedLabels[sentenceI] = Maybe.Just(RelationType.PER_EMPLOYEE_OF.canonicalName);
          }
          // Ensure the Y label has this annotated label
          if (rel.isDefined()) { positiveLabels.add(rel.get().canonicalName); }
        }
      }
    }
    // Get stats on positive labels
    for (String posLabel : positiveLabels) { assert RelationType.fromString(posLabel).orCrash().canonicalName.equals(posLabel); }

    return positiveLabels;
  }

  /**
   * Compute the negative relations to use for a give set of positive relations, and a sentence group.
   * @param positiveLabels The known positive relations for this group; see {@link KBPTrainer#cleanPositiveRelations(java.util.Set, edu.stanford.nlp.kbp.common.SentenceGroup, java.util.function.Function)}.
   * @param group The sentence group we are computing negative relations for.
   * @param querier The IR component to use -- this is used for accessing the knowledge base.
   * @return A clean set of negative relations to consider for this sentence group, given the positive relations known.
   */
  public static Set<String> computeNegativeRelations(Set<String> positiveLabels,
                                                     SentenceGroup group,
                                                     KBPIR querier) {
    KBPair key = group.key;
    Set<String> negativeLabels = new HashSet<>();
    // Get negative labels
    KBPEntity entity = key.getEntity();
    for (RelationType rel : RelationType.values()) {
      // Never add a negative that we know is a positive
      if (positiveLabels.contains(rel.canonicalName)) { continue; }
      if (Props.TRAIN_NEGATIVES_INCOMPLETE) {
        // Only add the negative label if we know of a positive slot fill
        // for the relation, and it's not this candidate slot fill.
        // For example, Barack Obama born_in Arizona should be added, since we know he was born in Hawaii
        // For safety, case insensitive matches are considered matches

        // Get the set of known slot fill values for this relation for this entity and check
        //  if they are potentially incompatible with the slot value that we are considering adding as a negative example
        // Check whether:
        //   1) we have any known slot fills  for this relation (first sign of potential incompatibility)
        //   2) one of the slot fill values for the relation matches the the one we are considering adding as negative example
        //      (then the slot value is compatible, so we don't want to add it to our negative examples)
        Set<String> knownSlotFills = querier.getKnownSlotValuesForEntityAndRelation(entity, rel);
        boolean relSlotFillsPotentiallyIncompatibleWithThisSlotValue = knownSlotFills.size() > 0;
        for (String knownFill : knownSlotFills) {
          if (Utils.sameSlotFill(key.slotValue, knownFill)) { relSlotFillsPotentiallyIncompatibleWithThisSlotValue = false; break; }
        }
        // Only add the negative label if the slot type is incompatible
        // That is, if the candidate negative label cannot co-occur with any of the positive labels,
        // add it as a negative example.
        boolean compatibleSlotType = true;
        if (Props.TRAIN_NEGATIVES_INCOMPATIBLE) {
          compatibleSlotType = positiveLabels.isEmpty();
          CHECK_COMPATIBLE: for (String positiveLabel : positiveLabels) {
            for (RelationType positiveRel : RelationType.fromString(positiveLabel)) {
              compatibleSlotType |= rel.plausiblyCooccursWith(positiveRel) || rel == positiveRel;
              // Good enough, break
              if (compatibleSlotType) break CHECK_COMPATIBLE;
            }
          }
        }
        // Add negative label, if we can
        // That is, either the relation is single valued and already has a known taken slot,
        // or it's incompatible to begin with.
        if ( (relSlotFillsPotentiallyIncompatibleWithThisSlotValue && rel.cardinality == RelationType.Cardinality.SINGLE) ||
            !compatibleSlotType) {
          negativeLabels.add(rel.canonicalName);
        }
      } else {
        // Add all negative relations, even if we can't confirm that they're negative
        negativeLabels.add(rel.canonicalName);
      }
    }
    if (Props.TRAIN_NEGATIVES_INCOMPLETE) {
      for (String positive : positiveLabels) { assert !negativeLabels.contains(positive); }
      for (String negative : negativeLabels) { assert !positiveLabels.contains(negative); }
    }
    negativeLabels.removeAll(positiveLabels);

    return negativeLabels;
  }

  /**
   * datums contain a list of sentence examples that might possibly express the KBTriple key.
   * This function aggregates all the possible labels between an entity pair and clusters together
   * the sentences. It then samples _some negative "relations"/labels_ and produces a dataset that
   * contains the entity pair, lists of positive and negative labels and the supporting sentences.
   * @param datums list of sentence examples.
   * @param positiveRelations The known positive labels, if we want to override the default.
   * @return A KBP Dataset
   */
  public KBPDataset<String, String> makeDataset(Iterator<SentenceGroup> datums,
                                                Maybe<Map<KBPair, Set<String>>> positiveRelations) {
    startTrack("Making dataset");
    logger.log("Train unlabeled = " + Props.TRAIN_UNLABELED + " with " + Props.TRAIN_UNLABELED_SELECT);

    // A whole lot of variables, debugging and otherwise
    KBPDataset<String, String> dataset = new KBPDataset<>();
    Random rand = new Random(0);
    int numDatumsWithPositiveLabels = 0;
    int numNegatives = 0;
    int numDatumsWithMultipleLabels = 0;
    int totalSentenceGroups = 0;
    int numAnnotatedDatumsAdded = 0;
    Set<String> allLabels = new HashSet<>();
    for (RelationType rel:RelationType.values()) {
      allLabels.add(rel.canonicalName);
    }

    // The actual loop starts here
    for( SentenceGroup group : new IterableIterator<>(datums)) {
      // Some accounting
      KBPair key = group.key;
      assert group.isValid();
      totalSentenceGroups += 1;
      if (totalSentenceGroups % 10000 == 0) {
        logger.log("read " + totalSentenceGroups + " sentence groups; [" + numDatumsWithPositiveLabels + " pos + " + numNegatives + " neg]; " + Utils.getMemoryUsage());
      }

      // Get all positive labels
      Set<String> positiveLabels = positiveRelations.isDefined() ? positiveRelations.get().get(key) : this.querier.getKnownRelationsForPair(key);
      if (positiveLabels == null) { positiveLabels = new HashSet<>(); }
      // Remove impossible positive labels (our KB is noisy...)
      if (Props.TRAIN_FIXKB) {
        Iterator<String> positiveLabelIter = positiveLabels.iterator();
        while (positiveLabelIter.hasNext()) {
          for (RelationType relation : RelationType.fromString(positiveLabelIter.next())) {
            if (relation.entityType != key.entityType || // if doesn't match entity type (e.g., a per: relation for an ORG)
                (key.slotType.isDefined() &&  // ... or doesn't match slot type (with fudge allowed for non-regexner tags, in case it hasn't been run)
                  key.slotType.get().isRegexNERType && !relation.validNamedEntityLabels.contains(key.slotType.get())) ) {
              positiveLabelIter.remove();  // remove it
            }
          }
        }
      }
      // Tweak employee_of and member_of (collapsed completely in 2013)
      if (positiveLabels.contains(RelationType.PER_MEMBER_OF.canonicalName)) {
        positiveLabels.remove(RelationType.PER_MEMBER_OF.canonicalName);
        positiveLabels.add(RelationType.PER_EMPLOYEE_OF.canonicalName);
      }
      // Augment positive labels with annotated examples
      @SuppressWarnings("unchecked")
      Maybe<String>[] annotatedLabels = new Maybe[group.size()];
      for (int sentenceI = 0; sentenceI < annotatedLabels.length; ++sentenceI) {
        // get annotated labels
        if (group.sentenceGlossKeys.isDefined()) {
          annotatedLabels[sentenceI] = Maybe.fromNull(annotationForSentence.get(group.sentenceGlossKeys.get().get(sentenceI)));
        } else {
          annotatedLabels[sentenceI] = Maybe.Nothing();
        }
        // If we have annotated
        if (annotatedLabels[sentenceI].isDefined()) {
          Maybe<RelationType> rel = RelationType.fromString(annotatedLabels[sentenceI].get());
          if (rel.isDefined() && rel.get().entityType != key.entityType) {
            // ignore label
            annotatedLabels[sentenceI] = Maybe.Nothing();
          } else {
            // Clean up employee / member distinction (these are common mistakes, and in fact collapsed in 2013)
            if (positiveLabels.contains(RelationType.PER_MEMBER_OF.canonicalName) && annotatedLabels[sentenceI].get().equals(RelationType.PER_EMPLOYEE_OF.canonicalName)) {
              positiveLabels.remove(RelationType.PER_MEMBER_OF.canonicalName);
              positiveLabels.add(RelationType.PER_EMPLOYEE_OF.canonicalName);
              if (Utils.assertionsEnabled()) { throw new AssertionError("Should already have prohibited " + RelationType.PER_MEMBER_OF.canonicalName + " as a positive label!"); }
            } else if (positiveLabels.contains(RelationType.PER_EMPLOYEE_OF.canonicalName) && annotatedLabels[sentenceI].get().equals(RelationType.PER_MEMBER_OF.canonicalName)) {
              annotatedLabels[sentenceI] = Maybe.Just(RelationType.PER_EMPLOYEE_OF.canonicalName);
            }
            // Ensure the Y label has this annotated label
            if (rel.isDefined()) { positiveLabels.add(rel.get().canonicalName); }
            numAnnotatedDatumsAdded += 1;
          }
        }
      }
      // Get stats on positive labels
      if (positiveLabels.size() > 0) { numDatumsWithPositiveLabels += 1; }
      if (positiveLabels.size() > 1) { numDatumsWithMultipleLabels += 1; }
      for (String posLabel : positiveLabels) { assert RelationType.fromString(posLabel).orCrash().canonicalName.equals(posLabel); }

      // Subsample negatives, trying to maintain a given ratio to all the positive labels.
      boolean addExample = positiveLabels.size() > 0;
      int expectedNumNegatives = (int) (((double) numDatumsWithPositiveLabels) * Props.TRAIN_NEGATIVES_RATIO);
      if (numNegatives < expectedNumNegatives) {  // if we need more negatives
        // Add if we subsample the example
        double skipProb = Math.pow(0.75, expectedNumNegatives - numNegatives);
        addExample |= rand.nextDouble() >= skipProb;  // add if it was subsampled
        // Don't add if it's a negative with >1 instance (more likely to be true a priori); and, we don't absolutely need to
        if (Props.TRAIN_NEGATIVES_SINGLETONS &&
            positiveLabels.size() == 0 && group.size() > 1 &&
            expectedNumNegatives - numNegatives > 100) {
          addExample = false;
        }
        // Add if the sentence group has annotations
        if (!addExample) {
          for (int sentI = 0; sentI < group.size(); ++sentI) {  // add if it's an annotated sentence
            if (annotationForSentence.containsKey(group.getSentenceGlossKey(sentI))) { addExample = true; break; }
          }
        }
      }

      Set<String> negativeLabels = new HashSet<>();
      if (addExample) {
        if (positiveLabels.size() == 0) { numNegatives += 1; }
        // Get negative labels
        KBPEntity entity = key.getEntity();
        for (RelationType rel : RelationType.values()) {
          // Never add a negative that we know is a positive
          if (positiveLabels.contains(rel.canonicalName)) { continue; }
          if (Props.TRAIN_NEGATIVES_INCOMPLETE) {
            // Only add the negative label if we know of a positive slot fill
            // for the relation, and it's not this candidate slot fill.
            // For example, Barack Obama born_in Arizona should be added, since we know he was born in Hawaii
            // For safety, case insensitive matches are considered matches

            // Get the set of known slot fill values for this relation for this entity and check
            //  if they are potentially incompatible with the slot value that we are considering adding as a negative example
            // Check whether:
            //   1) we have any known slot fills  for this relation (first sign of potential incompatibility)
            //   2) one of the slot fill values for the relation matches the the one we are considering adding as negative example
            //      (then the slot value is compatible, so we don't want to add it to our negative examples)
            Set<String> knownSlotFills = querier.getKnownSlotValuesForEntityAndRelation(entity, rel);
            boolean relSlotFillsPotentiallyIncompatibleWithThisSlotValue = knownSlotFills.size() > 0;
            for (String knownFill : knownSlotFills) {
              if (Utils.sameSlotFill(key.slotValue, knownFill)) { relSlotFillsPotentiallyIncompatibleWithThisSlotValue = false; break; }
            }
            // Only add the negative label if the slot type is incompatible
            // That is, if the candidate negative label cannot co-occur with any of the positive labels,
            // add it as a negative example.
            boolean compatibleSlotType = true;
            if (Props.TRAIN_NEGATIVES_INCOMPATIBLE) {
              compatibleSlotType = positiveLabels.isEmpty();
              CHECK_COMPATIBLE: for (String positiveLabel : positiveLabels) {
                for (RelationType positiveRel : RelationType.fromString(positiveLabel)) {
                  compatibleSlotType |= rel.plausiblyCooccursWith(positiveRel) || rel == positiveRel;
                  // Good enough, break
                  if (compatibleSlotType) break CHECK_COMPATIBLE;
                }
              }
            }
            // Add negative label, if we can
            // That is, either the relation is single valued and already has a known taken slot,
            // or it's incompatible to begin with.
            if ( (relSlotFillsPotentiallyIncompatibleWithThisSlotValue && rel.cardinality == RelationType.Cardinality.SINGLE) ||
                 !compatibleSlotType) {
              negativeLabels.add(rel.canonicalName);
            }
          } else {
            // Add all negative relations, even if we can't confirm that they're negative
            negativeLabels.add(rel.canonicalName);
          }
        }
        if (Props.TRAIN_NEGATIVES_INCOMPLETE) {
          for (String positive : positiveLabels) { assert !negativeLabels.contains(positive); }
          for (String negative : negativeLabels) { assert !positiveLabels.contains(negative); }
        }
        negativeLabels.removeAll(positiveLabels);
      }

      // Get unknown labels
      Set<String> unknownLabels = new HashSet<>();
      if (Props.TRAIN_UNLABELED) {
        switch (Props.TRAIN_UNLABELED_SELECT) {
          case NEGATIVE:
            unknownLabels.addAll(negativeLabels);
            break;
          case NOT_POSITIVE:
            unknownLabels.addAll(Sets.diff(allLabels, positiveLabels));
            break;
          case NOT_POSITIVE_WITH_NEGPOS:
            if (positiveLabels.size() > 0 || negativeLabels.size() > 0)
              unknownLabels.addAll(Sets.diff(allLabels, positiveLabels));
            break;
          case NOT_LABELED:
            unknownLabels.addAll(Sets.diff(Sets.diff(allLabels, positiveLabels), negativeLabels));
            break;
          case NOT_LABELED_WITH_NEGPOS:
            if (positiveLabels.size() > 0 || negativeLabels.size() > 0)
              unknownLabels.addAll(Sets.diff(Sets.diff(allLabels, positiveLabels), negativeLabels));
            break;
          default: throw new UnsupportedOperationException("Unsupported train.unlabeled.select " + Props.TRAIN_UNLABELED_SELECT);
        }
      }

      // Decide if we want to add this group
      // We only add the group only if we are going to use unlabeled data, or this group has positive or negative labels
      boolean addGroup = unknownLabels.size() > 0 || positiveLabels.size() > 0 || negativeLabels.size() > 0;

      // Add datum to dataset
      if (addGroup) {
        // Clean up coref, if applicable
        if (!Props.INDEX_COREF_DO) { group = group.filterFeature(FeatureFactory.COREF_FEATURE); }
        // Actually add the datum
        dataset.addDatum( positiveLabels, negativeLabels, unknownLabels, group, group.sentenceGlossKeys, annotatedLabels);
      }
    }

    // Post process dataset
    forceTrack("Applying feature count threshold (" + Props.FEATURE_COUNT_THRESHOLD + ")");
    dataset.applyFeatureCountThreshold(Props.FEATURE_COUNT_THRESHOLD);
    endTrack("Applying feature count threshold (" + Props.FEATURE_COUNT_THRESHOLD + ")");

    // Sanity Checks
    if (dataset.size() < numDatumsWithPositiveLabels) { throw new IllegalStateException("Fewer datums in dataset than in input"); }
    // Group size statistics
    int sumSentencesPerPositiveGroup = 0;
    int numPositiveGroups = 0;
    int sumSentencesPerNegativeGroup = 0;
    int numNegativeGroups = 0;
    int sumSentencesPerNoPosNoNegGroup = 0;
    for (int exI = 0; exI < dataset.size(); ++exI) {
      if (dataset.getPositiveLabels(exI).size() > 0) {
        numPositiveGroups += 1;
        sumSentencesPerPositiveGroup += dataset.getNumSentencesInGroup(exI);
      } else if (dataset.getNegativeLabels(exI).size() > 0) {
        // no positive, has negatives
        numNegativeGroups += 1;
        sumSentencesPerNegativeGroup += dataset.getNumSentencesInGroup(exI);
      } else {
        // no positive, no negatives, but still here - must just have unknown labels
        sumSentencesPerNoPosNoNegGroup += dataset.getNumSentencesInGroup(exI);
      }
    }
    // More Sanity Checks
    for (String label : dataset.labelIndex) {
      if (!label.equals(RelationType.fromString(label).orCrash().canonicalName)) {
        throw new IllegalStateException("Unknown relation label being added to dataset: " + label);
      }
    }

    // Print info
    startTrack("Dataset Info");
    logger.log(BLUE, "                                size: " + dataset.size());
    logger.log(BLUE, "           number of feature classes: " + dataset.numFeatures());
    logger.log(BLUE, "                 number of relations: " + dataset.numClasses());
    logger.log(BLUE, "                   datums in dataset: " + numPositiveGroups + " positive (" + numDatumsWithMultipleLabels + " with multiple relations); " + (dataset.size() - numPositiveGroups) + " negative groups");
    logger.log(BLUE, "           manually annotated datums: " + numAnnotatedDatumsAdded);
    logger.log(BLUE, " average sentences in positive group: " + (((double) sumSentencesPerPositiveGroup) / ((double) numPositiveGroups)));
    logger.log(BLUE, " average sentences in negative group: " + (((double) sumSentencesPerNegativeGroup) / ((double) numNegativeGroups)));
    logger.log(BLUE, "average sentences in unlabeled group: " + (((double) sumSentencesPerNoPosNoNegGroup) / ((double) (dataset.size() - numPositiveGroups - numNegativeGroups))));
    logger.log(BLUE, "          sentence groups considered: " + totalSentenceGroups);
    endTrack("Dataset Info");

    endTrack("Making dataset");
    return dataset;
  }

  public TrainingStatistics run() {
    // Train
    Pair<RelationClassifier, TrainingStatistics> statistics = trainOnTuples(querier.trainingTuples());
    // Save classifier
    this.classifier = statistics.first;
    try {
      logger.log(BOLD, BLUE, "saving model to " + Props.KBP_MODEL_PATH);
      classifier.save(Props.KBP_MODEL_PATH);
    } catch(IOException e) {
      logger.err("Could not save model.");
      logger.fatal(e);
    }
    // Save statistics
    try {
      IOUtils.writeObjectToFile(statistics.second,
          Props.WORK_DIR.getPath() + File.separator + "train_statistics.ser.gz");
    } catch (IOException e) {
      logger.err(e);
    }
    // Return
    return statistics.second;
  }

 }
