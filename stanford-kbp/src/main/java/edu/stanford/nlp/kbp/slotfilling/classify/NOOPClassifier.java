package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.common.SentenceGroup;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * A very simple NOOP classifier.
 *
 * @author Gabor Angeli
 */
public class NOOPClassifier extends RelationClassifier {
  @Override
  public Counter<Pair<String, Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentences) {
    return new ClassicCounter<>();
  }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> trainSet) {
    return TrainingStatistics.empty();
  }

  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    // Do nothing
  }

  @Override
  public void save(ObjectOutputStream out) throws IOException {
    // Do nothing
  }
}
