package edu.stanford.nlp.kbp.slotfilling.shallowdive;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.classify.KBPDataset;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.classify.TrainingStatistics;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer;
import edu.stanford.nlp.kbp.slotfilling.train.KryoDatumCache;
import edu.stanford.nlp.time.SUTimeSimpleParser;
import edu.stanford.nlp.util.Factory;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.sql.*;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A collection uf useful functions and tasks to do once a collection of datums is created.
 * Principally, this is training a classifier.
 *
 * @author Gabor Angeli
 */
public class DatumOps {

  private static Redwood.RedwoodChannels logger = Redwood.channels("DatumOps");

  public final KBPTrainer trainer;
  public final TextOps textOps;

  public DatumOps(TextOps textOps, KBPTrainer trainer) {
    this.textOps = textOps;
    this.trainer = trainer;
  }

  /**
   * Parse a Postgres row into a {@link edu.stanford.nlp.kbp.common.KBPair} corresponding to the datum's key.
   * @param rs The row of the Postgres table, as a cursor.
   * @return The {@link edu.stanford.nlp.kbp.common.KBPair} corresponding to the key for the datum represented
   * by this row.
   * @throws SQLException From the underlying Postgres implementation.
   */
  protected static KBPair mkKey(ResultSet rs) throws SQLException {
    return KBPNew.entName(rs.getString("entity_name"))
        .entType(rs.getString("entity_type"))
        .slotValue(rs.getString("slot_value"))
        .slotType(rs.getString("slot_value_type")).KBPair();
  }

  /**
   * Parse a (singleton) {@link edu.stanford.nlp.kbp.common.SentenceGroup} from this row of the datum table.
   * @param rs The row of the Postgres table, as a cursor.
   * @return The {@link edu.stanford.nlp.kbp.common.SentenceGroup} corresponding to this row of the table.
   * @throws SQLException From the underlying Postgres implementation.
   */
  protected static SentenceGroup mkDatum(ResultSet rs) throws SQLException {
    try {
      ByteArrayInputStream is = new ByteArrayInputStream(rs.getBytes("datum"));
      SentenceGroup g =  KryoDatumCache.load(is);
      is.close();
      return g;
    } catch (IOException e) {
      logger.log(e);
      return SentenceGroup.empty(mkKey(rs));
    } catch (ClassNotFoundException e) {
      logger.log(e);
      return SentenceGroup.empty(mkKey(rs));
    }
  }


  /**
   * A little temporary class to store datum info coming out of the database...
   */
  private static class KBPDatum {
    public final SentenceGroup group;
    public Set<String> positiveLabels = new HashSet<>();
    public Set<String> negativeLabels = new HashSet<>();
    public Set<String> unknownLabels  = new HashSet<>();
    public KBPDatum(SentenceGroup group) {
      this.group = group;
    }
    public void registerRelation(boolean truth, String relationName) {
      if (truth) {
        positiveLabels.add(relationName);
      } else {
        negativeLabels.add(relationName);
      }
    }
  }

  @SuppressWarnings("unchecked")
  public KBPDataset<String, String> mkDataset(String datumTable) {
    Pointer<IterableIterator<KBPDatum>> datumIterator = new Pointer<>();

    PostgresUtils.withConnection(datumTable, psql -> {
      // Set up statement
      Statement stmt = psql.createStatement();
      final ResultSet cursor = stmt.executeQuery(
          "SELECT d.did, d.entity_name, d.entity_type, d.slot_value, d.slot_value_type, d.datum, r.relation_name, r.truth FROM " +
          datumTable + " d, " + (datumTable + "_relations") + " r WHERE " +
          "d.did = r.did " +
          "ORDER BY d.entity_name, d.entity_type, d.slot_value, d.slot_value_type, r.truth DESC;"  // DESC is important
      );
      final boolean savedAutoCommit = psql.getAutoCommit();
      final Random rand = new Random(42);
      psql.setAutoCommit(false);
      stmt.setFetchSize(100000);
      if (!cursor.next()) {
        throw new IllegalArgumentException("No datums in table");
      }

      // Read data as iterator
      datumIterator.set(new IterableIterator<>(CollectionUtils.iteratorFromMaybeFactory(new Factory<Maybe<KBPDatum>>() {
        private KBPair currentKey = mkKey(cursor);
        private KBPDatum currentDatum = new KBPDatum(SentenceGroup.empty(currentKey));
        /** A hack to avoid duplicating datums; partiuclarly, to avoid reading duplicated datums */
        private Set<Long> didsSeen = new HashSet<>();
        private boolean isDone = false;
        private int numPositives = 0;
        private int numNegatives = 0;

        @Override
        public Maybe<KBPDatum> create() {
          // INPUT ASSUMPTION:
          //   Every call to this method assumes that there is a valid SentenceGroup waiting on the cursor.
          //   If we are at the end of the stream, then isDone should be set to true.
          try {
            // Initialize candidate group
            if (isDone) {
              return null;
            }
            // Subsample
            boolean hasTrueRelation = cursor.getBoolean("truth");
            boolean keepSample = hasTrueRelation;
            if (!hasTrueRelation) {
              int expectedNumNegatives = (int) (((double) numPositives) * Props.TRAIN_NEGATIVES_RATIO);
              if (numNegatives < expectedNumNegatives) {  // if we need more negatives
                // Add if we subsample the example
                double skipProb = Math.pow(0.75, expectedNumNegatives - numNegatives);
                keepSample = rand.nextDouble() >= skipProb;
              }
            }
            if ((numPositives + numNegatives) % 100000 == 0)
              logger.log("read " + (numPositives + numNegatives) + " sentence groups; [" + numPositives + " pos + " + numNegatives + " neg]; " + Utils.getMemoryUsage());
            // Read
            if (keepSample) {
              // Keep the sample
              // (update count)
              if (hasTrueRelation) {
                numPositives += 1;
              } else {
                numNegatives += 1;
              }
              // (update group)
              didsSeen.add(cursor.getLong("did"));
              currentDatum.group.merge(mkDatum(cursor));
              currentDatum.registerRelation(cursor.getBoolean("truth"), cursor.getString("relation_name"));
              while (cursor.next()) {
                KBPair key = mkKey(cursor);
                if (key.equals(currentKey)) {
                  long did = cursor.getLong("did");
                  if (!didsSeen.contains(did)) {
                    currentDatum.group.merge(mkDatum(cursor));
                    didsSeen.add(did);
                  }
                  currentDatum.registerRelation(cursor.getBoolean("truth"), cursor.getString("relation_name"));
                } else {
                  currentKey = key;
                  didsSeen.clear();
                  return Maybe.Just(currentDatum);
                }
              }
            } else {
              // Loop over this sample
              while (cursor.next()) {
                KBPair key = mkKey(cursor);
                if (!key.equals(currentKey)) {
                  currentKey = key;
                  didsSeen.clear();
                  return Maybe.<KBPDatum>Nothing();
                }
              }
            }
            // If we reached here, there's no more datums
            isDone = true;
            if (currentDatum.group.isEmpty()) {
              return Maybe.<KBPDatum>Nothing();
            } else {
              return Maybe.Just(currentDatum);
            }
          } catch (SQLException e) {
            throw new RuntimeException(e);
          }
        }
      })));

      // Restore PSQl
      if (savedAutoCommit != psql.getAutoCommit()) {
        psql.setAutoCommit(savedAutoCommit);
      }
    });

    // Create dataset from iterator
    forceTrack("Creating dataset");
    KBPDataset<String,String> dataset = new KBPDataset<>();
    for (KBPDatum datum : datumIterator.dereference().get()) {
      Maybe<String>[] annotatedLabels = new Maybe[datum.group.size()];
      dataset.addDatum( datum.positiveLabels, datum.negativeLabels, datum.unknownLabels, datum.group, datum.group.sentenceGlossKeys, annotatedLabels );
    }
    dataset.applyFeatureCountThreshold(Props.FEATURE_COUNT_THRESHOLD);
    startTrack("Dataset Info");
    logger.log(BLUE, "                                size: " + dataset.size());
    logger.log(BLUE, "           number of feature classes: " + dataset.numFeatures());
    logger.log(BLUE, "                 number of relations: " + dataset.numClasses());
    endTrack("Dataset Info");
    endTrack("Creating Dataset");
    return dataset;
  }

  /**
   * Train a model. This is effectively a wrapper around
   * {@link edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer#trainOnData(edu.stanford.nlp.kbp.slotfilling.classify.KBPDataset)}.
   * This function should also implement the other tasks in
   * {@link edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer#run()}/
   *
   * @param tableName The table with the input datums.
   * @param ir The IR component, used to retrieve the training triples.
   * @return A relation classifier, and the statistics from training.
   */
  public Pair<RelationClassifier, TrainingStatistics> train(String tableName, KBPIR ir) {
    // Train
    forceTrack("Training");
    Pair<RelationClassifier, TrainingStatistics> statistics = trainer.trainOnData(mkDataset(tableName));
    // Save classifier
    try {
      logger.log(BOLD, BLUE, "saving model to " + Props.KBP_MODEL_PATH);
      statistics.first.save(Props.KBP_MODEL_PATH);
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
    endTrack("Training");
    // Return
    return statistics;
  }
}
