package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.classify.EnsembleRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.KBPDataset;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.classify.TrainingStatistics;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.common.Props;
import edu.stanford.nlp.kbp.common.RelationType;
import edu.stanford.nlp.kbp.common.SentenceGroup;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import org.junit.*;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import static org.junit.Assert.*;

/**
 * A basic test to make sure the Ensemble Relation Extractor is doing something reasonable
 *
 * @author Gabor Angeli
 */
public class EnsembleRelationExtractorTest {

  public static class AlwaysGuessOneRelationClassifier extends RelationClassifier {
    public final RelationType relationToGuess;
    public final double confidenceToGuessAt;

    public AlwaysGuessOneRelationClassifier(RelationType relationToGuess, double confidenceToGuessAt) {
      this.relationToGuess = relationToGuess;
      this.confidenceToGuessAt = confidenceToGuessAt;
    }

    public AlwaysGuessOneRelationClassifier(RelationType relationToGuess) {
      this(relationToGuess, 1.0);
    }

    @Override
    public TrainingStatistics train(KBPDataset<String, String> trainSet) {
      return TrainingStatistics.empty();
    }
    @Override
    public void load(ObjectInputStream in) throws IOException, ClassNotFoundException { }
    @Override
    public void save(ObjectOutputStream out) throws IOException { }
    @Override
    public Pair<Double, Maybe<KBPRelationProvenance>> classifyRelation(SentenceGroup input, RelationType relation, Maybe<CoreMap[]> rawSentences) {
      if (relation == relationToGuess) {
        return Pair.makePair(confidenceToGuessAt, Maybe.<KBPRelationProvenance>Nothing());
      } else {
        return Pair.makePair(0.0, Maybe.<KBPRelationProvenance>Nothing());
      }
    }
  }

  private EnsembleRelationExtractor ensemble = null;
  private EnsembleRelationExtractor ensembleInAgreement = null;

  @Before
  public void createEnsembleClassifier() {
    this.ensemble = new EnsembleRelationExtractor(
        new AlwaysGuessOneRelationClassifier(RelationType.PER_CITY_OF_BIRTH),
        new AlwaysGuessOneRelationClassifier(RelationType.PER_CITY_OF_BIRTH),
        new AlwaysGuessOneRelationClassifier(RelationType.PER_CITY_OF_BIRTH),
        new AlwaysGuessOneRelationClassifier(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH),
        new AlwaysGuessOneRelationClassifier(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH),
        new AlwaysGuessOneRelationClassifier(RelationType.PER_CITY_OF_DEATH)
    );
    this.ensembleInAgreement = new EnsembleRelationExtractor(
        new AlwaysGuessOneRelationClassifier(RelationType.PER_CITY_OF_BIRTH),
        new AlwaysGuessOneRelationClassifier(RelationType.PER_CITY_OF_BIRTH),
        new AlwaysGuessOneRelationClassifier(RelationType.PER_CITY_OF_BIRTH)
    );

  }


  @Test
  public void testClassifyStrategyAny() {
    Props.TEST_ENSEMBLE_COMBINATION = Props.EnsembleCombinationMethod.AGREE_ANY;
    Counter<String> predictions = ensemble.classifyRelationsNoProvenance(null, Maybe.<CoreMap[]>Nothing());
    assertTrue(predictions.containsKey(RelationType.PER_CITY_OF_BIRTH.canonicalName));
    assertEquals(1.0, predictions.getCount(RelationType.PER_CITY_OF_BIRTH.canonicalName), 1e-5);
    assertTrue(predictions.containsKey(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName));
    assertEquals(1.0 , predictions.getCount(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName), 1e-5);
    assertTrue(predictions.containsKey(RelationType.PER_CITY_OF_DEATH.canonicalName));
    assertEquals(1.0, predictions.getCount(RelationType.PER_CITY_OF_DEATH.canonicalName), 1e-5);

    predictions = ensembleInAgreement.classifyRelationsNoProvenance(null, Maybe.<CoreMap[]>Nothing());
    assertTrue(predictions.containsKey(RelationType.PER_CITY_OF_BIRTH.canonicalName));
    assertEquals(1.0, predictions.getCount(RelationType.PER_CITY_OF_BIRTH.canonicalName), 1e-5);
  }

  @Test
  public void testClassifyStrategyMost() {
    Props.TEST_ENSEMBLE_COMBINATION = Props.EnsembleCombinationMethod.AGREE_MOST;
    Counter<String> predictions = ensemble.classifyRelationsNoProvenance(null, Maybe.<CoreMap[]>Nothing());
    assertTrue(predictions.containsKey(RelationType.PER_CITY_OF_BIRTH.canonicalName));
    assertEquals(1.0, predictions.getCount(RelationType.PER_CITY_OF_BIRTH.canonicalName), 1e-5);
    assertFalse(predictions.containsKey(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName));
    assertEquals(0.0, predictions.getCount(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName), 1e-5);
    assertFalse(predictions.containsKey(RelationType.PER_CITY_OF_DEATH.canonicalName));
    assertEquals(0.0, predictions.getCount(RelationType.PER_CITY_OF_DEATH.canonicalName), 1e-5);

    predictions = ensembleInAgreement.classifyRelationsNoProvenance(null, Maybe.<CoreMap[]>Nothing());
    assertTrue(predictions.containsKey(RelationType.PER_CITY_OF_BIRTH.canonicalName));
    assertEquals(1.0, predictions.getCount(RelationType.PER_CITY_OF_BIRTH.canonicalName), 1e-5);
  }

  @Test
  public void testClassifyStrategyAll() {
    Props.TEST_ENSEMBLE_COMBINATION = Props.EnsembleCombinationMethod.AGREE_ALL;
    Counter<String> predictions = ensemble.classifyRelationsNoProvenance(null, Maybe.<CoreMap[]>Nothing());
    assertFalse(predictions.containsKey(RelationType.PER_CITY_OF_BIRTH.canonicalName));
    assertEquals(0.0, predictions.getCount(RelationType.PER_CITY_OF_BIRTH.canonicalName), 1e-5);
    assertFalse(predictions.containsKey(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName));
    assertEquals(0.0, predictions.getCount(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName), 1e-5);
    assertFalse(predictions.containsKey(RelationType.PER_CITY_OF_DEATH.canonicalName));
    assertEquals(0.0, predictions.getCount(RelationType.PER_CITY_OF_DEATH.canonicalName), 1e-5);

    predictions = ensembleInAgreement.classifyRelationsNoProvenance(null, Maybe.<CoreMap[]>Nothing());
    assertTrue(predictions.containsKey(RelationType.PER_CITY_OF_BIRTH.canonicalName));
    assertEquals(1.0, predictions.getCount(RelationType.PER_CITY_OF_BIRTH.canonicalName), 1e-5);
  }

}
