package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.util.MetaClass;

import java.io.IOException;
import java.util.Properties;

import static edu.stanford.nlp.util.logging.Redwood.Util.fatal;
import static edu.stanford.nlp.util.logging.Redwood.Util.log;

/**
 * <p>A collection of models which can be used to train the {@link RelationClassifier} component of the KBP system.
 * Many of these are here for historical reasons, but some deserve special mention:</p>
 *
 * <ul>
 *   <li>{@link ModelType#JOINT_BAYES}: The MIML-RE model of http://nlp.stanford.edu/pubs/emnlp2012-mimlre.pdf</li>
 *   <li>{@link ModelType#LOCAL_BAYES}: The distantly-supervised variant of MIML-RE -- roughly the model of Mintz et al.
 *   <li>{@link ModelType#LR_INC} / {@link ModelType#ROBUST_LR}: The model used in 2011, and that same model used with Julie's robust logistic regression.</li>
 *   <li>{@link ModelType#SUPERVISED}: A supervised relation extractor, making use of annotated datums loaded from {@link edu.stanford.nlp.kbp.common.Props#TRAIN_ANNOTATED_SENTENCES_DATA}</li>
 *   <li>{@link ModelType#ENSEMBLE}: An ensemble relation extractor, useful for model combination, and particularly for sampling examples for active learning. </li>
 * </ul>
 *
 * <p>Each element in the enum is parameterized by the class it is loading, and the constructor arguments it should pass, as well as some
 * metadata about the model. Each model is then constructed using a constructor taking a Properties file, and the optional additional
 * passed arguments</p>
 * 
 * <p> The first argument is a name, the second is whether it's locally trained (i.e., not MIML-RE), the third argument is the class to create,
 *  and the fourth argument is the arguments to pass into the constructor of the class </p>
 */
public enum ModelType {
  NOOP              ("noop",              true,  NOOPClassifier.class,                  new Object[]{}),     // A model performing a NOOP (never prediction a relation)
  LR_INC            ("lr_inc",            true,  OneVsAllRelationExtractor.class,       new Object[]{}),     // LR with incomplete information (used at KBP 2011)
  PERCEPTRON        ("perceptron",        false, PerceptronExtractor.class,             new Object[]{}),     // boring local Perceptron
  PERCEPTRON_INC    ("perceptron_inc",    false, PerceptronExtractor.class,             new Object[]{}),     // local Perceptron with incomplete negatives
  AT_LEAST_ONCE     ("at_least_once",     false, HoffmannExtractor.class,               new Object[]{}),     // (Hoffman et al, 2011)
  AT_LEAST_ONCE_INC ("at_least_once_inc", false, PerceptronExtractor.class,             new Object[]{}),     // AT_LEAST_ONCE with incomplete information
  LOCAL_BAYES       ("local_bayes",       false, JointBayesRelationExtractor.class,     new Object[]{true}), // Mintz++
  JOINT_BAYES       ("joint_bayes",       false, JointBayesRelationExtractor.class,     new Object[]{}),     // MIML-RE
  ROBUST_LR         ("robust_lr",         true,  OneVsAllRelationExtractor.class,       new Object[]{true}), // robust LR with shift parameters
  ENSEMBLE          ("ensemble",          false, EnsembleRelationExtractor.class,       new Object[]{}),     // ensemble classifier
  TOKENSREGEX       ("tokensregex",       true,  TokensRegexExtractor.class,            new Object[]{}),     // heuristic token regex extractor
  SEMGREX           ("semgrex",           true,  SemgrexExtractor.class,                new Object[]{}),
  TOP_EMPLOYEE      ("top_employee",      true,  TopEmployeesClassifier.class,          new Object[]{}),     // silly classifier for top employees
  SUPERVISED        ("supervised",        true,  SupervisedExtractor.class,             new Object[]{}),     // supervised relation extractor, using annotated labels

  // Not real models, but rather just for debugging
  GOLD              ("gold",              false, GoldClassifier.class,                  new Object[]{});     // memorize the data

  public final String name;
  public final boolean isLocallyTrained;
  public final Class<? extends RelationClassifier> modelClass;
  public final Object[] constructorArgs;

  ModelType(String name, boolean isLocallyTrained, Class<? extends RelationClassifier> modelClass, Object[] constructorArgs) {
    this.name = name;
    this.isLocallyTrained = isLocallyTrained;
    this.modelClass = modelClass;
    this.constructorArgs = constructorArgs;
  }

  /**
   * Construct a new model of this type.
   * @param props The properties file to use in the construction of the model.
   * @param <A> The type of the model being constructed. The user must make sure this type is valid.
   * @return A new model of the type specified by this ModelType.
   */
  public <A extends RelationClassifier> A construct(Properties props) {
    // Create MetaClass for loading
    log("constructing new model of type " + modelClass.getSimpleName());
    MetaClass clazz = new MetaClass(modelClass);
    // Create arguments
    Object[] args = new Object[constructorArgs.length + 1];
    args[0] = props;
    System.arraycopy(constructorArgs, 0, args, 1, constructorArgs.length);
    // Create instance
    try {
      return clazz.createInstance(args);
    } catch (MetaClass.ConstructorNotFoundException e) {
      fatal("classifier of type " + modelClass.getSimpleName() + " has no constructor "  + modelClass.getSimpleName() + "(Properties, ...)");
    }
    throw new IllegalStateException("code cannot reach here");
  }

  /**
   * Load a model of this type from a path.
   * @param path The path to the model being loaded.
   * @param props The properties file to use when loading the model.
   * @param <A> The type of the model being constructed. The user must make sure this type is valid.
   * @return The [presumably trained] model specified by the path (and properties).
   */
  @SuppressWarnings("unchecked")
  public <A extends RelationClassifier> A load(String path, Properties props) {
    assert path != null;
    log("loading model of type " + modelClass.getSimpleName() + " from " + path);
    try {
      // Route this call to Load to a call of AbstractModel.load
      return RelationClassifier.load(path, props, (Class<A>) modelClass);
    } catch (IOException e) {
      fatal("IOException while loading model of type: " + modelClass.getSimpleName() + " at " + path);
    } catch (ClassNotFoundException e) {
      fatal("Could not find class: " + modelClass.getSimpleName() + " at " + path);
    }
    throw new IllegalStateException("code cannot reach here");
  }

  public static Maybe<ModelType> fromString(String name) {
    for (ModelType slot : ModelType.values()) {
      if (slot.name.equals(name)) return Maybe.Just(slot);
    }
    return Maybe.Nothing();
  }
}
