package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.classify.*;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.common.Props;
import edu.stanford.nlp.kbp.common.RelationType;
import edu.stanford.nlp.kbp.common.SentenceGroup;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Properties;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A simple supervised classifier trained on only annotated labels.
 *
 * @author Gabor Angeli
 */
public class SupervisedExtractor extends RelationClassifier {
  private LinearClassifier<String, String> classifier;
  private boolean trained = false;

  private Index<String> labelIndexOrNull = null;
  private Index<String> featureIndexOrNull = null;

  @SuppressWarnings("UnusedParameters")
  public SupervisedExtractor(Properties ignored) { }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> trainSet) {
    trained = true;
    // Create dataset
    WeightedDataset<String, String> dataset = new WeightedDataset<>();
    if (labelIndexOrNull != null) { dataset.labelIndex = labelIndexOrNull; }
    if (featureIndexOrNull != null) { dataset.featureIndex = featureIndexOrNull; }
    Counter<RelationType> empiricalDistribution = new ClassicCounter<>();
    for (int groupI = 0; groupI < trainSet.size(); ++groupI) {
      List<Datum<String, String>> datumsInGroup = trainSet.getDatumGroup(groupI);
      for (int sentI = 0; sentI < trainSet.getNumSentencesInGroup(groupI); ++sentI) {
        Maybe<String> annotatedLabel = trainSet.getAnnotatedLabels(groupI)[sentI];
        for (String y : annotatedLabel) {
          if (!y.equals(RelationMention.UNRELATED)) { empiricalDistribution.incrementCount(RelationType.fromString(y).orCrash()); }
          dataset.add(new BasicDatum<>(datumsInGroup.get(sentI).asFeatures(), y));
        }
      }
    }
    log(BLUE, "training on " + dataset.size() + " annotated datums");

    // Re-normalize the datum weights to the empirical distribution
    Counters.normalize(empiricalDistribution);
    if (Props.TRAIN_SUPERVISED_REWEIGHT) {
      Counter<RelationType> datumWeights = Counters.division(RelationType.priors, empiricalDistribution);
      startTrack("Scaling factors");
      for (Pair<RelationType, Double> entry : Counters.toDescendingMagnitudeSortedListWithCounts(datumWeights)) {
        log(new DecimalFormat("0.00000").format(entry.second) + "\t" + entry.first);
      }
      endTrack("Scaling factors");
      for (int i = 0; i < dataset.size(); ++i) {
        for (RelationType guessedLabel : RelationType.fromString(dataset.getDatum(i).label())) {
          dataset.setWeight(i, (float) datumWeights.getCount(guessedLabel));
        }
      }
    }

    // Train classifier
    LinearClassifierFactory<String, String> factory = new LinearClassifierFactory<>(1e-4, false, Props.TRAIN_JOINTBAYES_ZSIGMA);
    switch (Props.TRAIN_JOINTBAYES_ZMINIMIZER) {
      case SGD: factory.useInPlaceStochasticGradientDescent(75, 1000, Props.TRAIN_JOINTBAYES_ZSIGMA); break;
      case SGDTOQN: factory.useHybridMinimizerWithInPlaceSGD(10, 1000, Props.TRAIN_JOINTBAYES_ZSIGMA); break;
      case QN: break; // default
    }
    this.classifier = factory.trainClassifier(dataset);
    return TrainingStatistics.empty();
  }

  @Override
  public Counter<Pair<String, Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentences) {
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> predictions = new ClassicCounter<>();

    for (int datumI = 0; datumI < input.size(); ++datumI) {
      Counter<String> relations = classifier.probabilityOf(input.get(datumI));
      String prediction = Counters.argmax(relations);
      if (!prediction.equals(RelationMention.UNRELATED) && relations.getCount(prediction) > Props.TRAIN_SUPERVISED_THRESHOLD) {
        double probability = relations.getCount(prediction);
        KBPRelationProvenance provenance = input.getProvenance(datumI);
        predictions.incrementCount(Pair.makePair(prediction, Maybe.Just(provenance)), probability);
      }
    }

    return predictions;
  }

  /**
   * Return this supervised relation extractor as a raw classifier.
   * If it is trained, the datasest parameter may be omitted; otherwise, it will train the extractor
   * fro mthe provided dataset.
   *
   * @param dataset An optional dataset to train the relation extractor with.
   * @return A JavaNLP classifier corresponding to this supervised relation extractor.
   */
  public LinearClassifier<String, String> asClassifier(Maybe<KBPDataset<String,String>> dataset) {
    if (!trained && dataset.isDefined()) {
      train(dataset.get());
      return classifier;
    } else if (trained) {
      return classifier;
    } else {
      throw new IllegalArgumentException("This classifier is not trained, and no dataset was provided");
    }
  }

  /**
   * Override the default indices used during training with custom indices.
   * This is primarily useful if used within another model (currently, {@link JointBayesRelationExtractor}),
   * where one would like to maintain the same indices throughout the model.
   *
   * @param labelIndex The label index to override, or null to keep the default index.
   * @param featureIndex The feature index to override, or null to keep the default index.
   * @return This same relation extractor, mutated with the new indices.
   */
  public SupervisedExtractor setIndices(Index<String> labelIndex, Index<String> featureIndex) {
    this.labelIndexOrNull = labelIndex;
    this.featureIndexOrNull = featureIndex;
    return this;
  }

  @SuppressWarnings("unchecked")
  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    this.classifier = (LinearClassifier<String, String>) in.readObject();
    trained = true;
  }

  @Override
  public void save(ObjectOutputStream out) throws IOException {
    out.writeObject(classifier);
  }
}
