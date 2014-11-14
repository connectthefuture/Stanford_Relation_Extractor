package edu.stanford.nlp.kbp.slotfilling;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import com.typesafe.config.ConfigParseOptions;
import com.typesafe.config.ConfigValue;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.classify.HackyModelCombination;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPEvaluator;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPScore;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPSlotValidator;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer;
import edu.stanford.nlp.util.Execution;

import java.io.File;
import java.util.function.Function;

import edu.stanford.nlp.util.MetaClass;
import edu.stanford.nlp.util.SendMail;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.IOException;
import java.util.Calendar;
import java.util.Date;
import java.util.Map;
import java.util.Properties;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * This is the entry point for training and testing the KBP Slotfilling system on the official evaluations.
 *
 * @author Gabor Angeli
 */
public class SlotfillingSystem {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("MAIN");

  public final Properties props;

  public SlotfillingSystem(Properties props) {
    this.props = props;
    if (props.getProperty("annotators") == null) {
      props.setProperty("annotators", Props.ANNOTATORS);
    }
  }


  //
  // Dependency Graph
  //
  // ir             classify
  //  |-> process---    |
  //  |     v      |    |
  //  |-> train <-------|
  //  |            v    |
  //  ------> evaluate <-
  //

  //
  // IR Component
  //
  public Lazy<KBPIR> ir = new Lazy<KBPIR>() {
    @Override
    protected KBPIR compute() {
      // If cached, return
      //noinspection LoopStatementThatDoesntLoop
      forceTrack("Creating IR");
      // Create new querier
      KBPIR querier = MetaClass.create(Props.INDEX_CLASS).createInstance(props);
      // Return
      endTrack("Creating IR");
      return querier;
    }
  };
  public KBPIR getIR() { return ir.get(); }

  //
  // Process Component
  //
  public Lazy<KBPProcess> process = new Lazy<KBPProcess>() {
    @Override
    protected KBPProcess compute() {
      forceTrack("Creating Process");
      // Create new processor
      KBPProcess process = new KBPProcess(props, ir);
      // Return
      endTrack("Creating Process");
      return process;
    }
  };
  public synchronized KBPProcess getProcess() { return process.get(); }

  //
  // Classify Component
  //
  private Maybe<Lazy<RelationClassifier>> classifier = Maybe.Nothing();
  public synchronized Lazy<RelationClassifier> getClassifier(final Maybe<String> modelFilename) {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (Lazy<RelationClassifier> m : this.classifier) { return m; }
    forceTrack("Creating Classifier");
    // Create new classifier
    Lazy<RelationClassifier> classifier;
    if (modelFilename.isDefined()){
      startTrack("Loading Classifier");
      if (Props.HACKS_HACKYMODELCOMBINATION) {
        classifier = new Lazy<RelationClassifier>() {
          @Override
          protected RelationClassifier compute() {
            return new HackyModelCombination(props);
          }
        };
      } else {
        classifier = new Lazy<RelationClassifier>() {
          @Override
          protected RelationClassifier compute() {
            return Props.TRAIN_MODEL.load(modelFilename.get(), props);
          }
        };
      }
      endTrack("Loading Classifier");
    } else {
      startTrack("Constructing Classifier");
      classifier = new Lazy<RelationClassifier>() {
        @Override
        protected RelationClassifier compute() {
          return Props.TRAIN_MODEL.construct(props);
        }
      };
      endTrack("Constructing Classifier");
    }
    this.classifier = Maybe.Just(classifier);
    // Return
    endTrack("Creating Classifier");
    return this.classifier.get();
  }

  /** Create a new classifier */
  public Lazy<RelationClassifier> getNewClassifier() { return getClassifier(Maybe.<String>Nothing()); }
  /** Load an existing classifier */
  public Lazy<RelationClassifier> getTrainedClassifier() { return getClassifier(Maybe.Just(Props.KBP_MODEL_PATH)); }

  //
  // Training Component
  //
  private Maybe<edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer> trainer = Maybe.Nothing();
  public KBPTrainer getTrainer() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer t : this.trainer) { return t; }
    forceTrack("Creating Trainer");
    // Create new trainer
    this.trainer = Maybe.Just(new KBPTrainer(getIR(), getProcess(), Props.TRAIN_MODEL.construct(props)));
    // Return
    endTrack("Creating Trainer");
    return this.trainer.get();
  }

  //
  // Evaluation Component
  //
  private Maybe<KBPEvaluator> evaluator = Maybe.Nothing();
  public synchronized KBPEvaluator getEvaluator() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (KBPEvaluator e : this.evaluator) { return e; }
    forceTrack("Creating Evaluator");
    // Create new evaluator
    this.evaluator = Maybe.Just(new KBPEvaluator(props, ir, process, getTrainedClassifier()));
    // Return
    endTrack("Creating Evaluator");
    return this.evaluator.get();
  }

  //
  // Evaluation Component
  //
  private Maybe<KBPSlotValidator> validator = Maybe.Nothing();
  public synchronized KBPSlotValidator getSlotValidator() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (KBPSlotValidator e : this.validator) { return e; }
    forceTrack("Creating Validator");
    // Create new evaluator
    this.validator = Maybe.Just(new KBPSlotValidator(props, getIR(), getProcess(), getTrainedClassifier().get()));
    // Return
    endTrack("Creating Validator");
    return this.validator.get();
  }

  /**
   * A utility method for various common tasks one may wish to perform with
   * a slotfilling system, but which are not part of the core functionaility and in general
   * don't depend on the KBP infrastructure (for the latter, calling methods in the various
   * components is preferred.
   *
   * @return A utility class from which many useful methods can be called.
   */
  public SlotfillingTasks go() { return new SlotfillingTasks(this); }


  /**
   * The main operation to do when calling SlotfillingSystem
   */
  public static enum RunMode { TRAIN_ONLY, EVALUATE_ONLY, TRAIN_AND_EVALUATE, VALIDATE, CONSOLE, DO_NOTHING }

  /**
   * A main router to various modes of running the program
   * @param mode The mode to run the program in
   * @param props The properties to run the program with
   * @throws Exception If something goes wrong
   */
  public static void runProgram(RunMode mode, Properties props) throws Exception {
    // Collect data
    Date startTime = Calendar.getInstance().getTime();
    Maybe<KBPScore> score = Maybe.Nothing();
    Maybe<Throwable> error = Maybe.Nothing();

    // Run program
    try {
      logger.log(FORCE, BOLD, BLUE, "run mode: " + mode);
      SlotfillingSystem instance = new SlotfillingSystem(props);
      boolean evaluate = true;
      switch(mode){
        case TRAIN_ONLY:
          evaluate = false;
        case TRAIN_AND_EVALUATE:
          instance.getTrainer().run();
        case EVALUATE_ONLY:
          if (evaluate) {
            instance.classifier = Maybe.Nothing(); // clear old classifier
            score = score.orElse(instance.getEvaluator().run());
          }
          break;
        case VALIDATE:
          instance.getSlotValidator().run();
          break;
        case CONSOLE:
          instance.go().console();
          break;
        default:
          logger.fatal("Unknown run mode: " + mode);
      }
    } catch (Throwable t) {
      error = Maybe.Just(t);
    }

    // Report Data (via email)
    for (String address : Props.KBP_EMAIL) {
      SendMail.sendHTMLMail(address, Utils.mkEmailSubject(score, error), Utils.mkEmailBody(score, error, startTime));
    }
    // Pass along any encountered exceptions
    //noinspection LoopStatementThatDoesntLoop
    for (Throwable t : error) { if (t instanceof RuntimeException) { throw (RuntimeException) t; } else { throw new RuntimeException(t); } }
  }

  public static void exec(Runnable toRun, Config options) {
    exec(toRun, options, false);
  }
  public static void exec(Runnable toRun, Config options, boolean exit) {
    Properties props = new Properties();
    for (Map.Entry<String, ConfigValue> entry : options.entrySet()) {
      props.put(entry.getKey(), entry.getValue().unwrapped());
    }
    Execution.exec(toRun, props, exit);
  }

  /**
   * A central method which takes command line arguments, and starts a program.
   * This method handles parsing the command line arguments, and setting the options in Props,
   * and any options from calling classes (working up the stack trace).
   * @param toRun The function to run, containing the implementation of the program
   * @param args The command line arguments passed to the program
   */
  public static void exec(final Function<Properties, Object> toRun, String[] args) {
    // Set options classes
    StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    Execution.optionClasses = new Class<?>[stackTrace.length +1];
    Execution.optionClasses[0] = Props.class;
    for (int i=0; i<stackTrace.length; ++i) {
      try {
        Execution.optionClasses[i+1] = Class.forName(stackTrace[i].getClassName());
      } catch (ClassNotFoundException e) {  // it's exceptions like these that make me hate Java.
        Execution.optionClasses[i+1] = SlotfillingSystem.class;
        logger.err(e);
      }
    }
    // Start Program
    if (args.length == 1) {
      // Case: Run with TypeSafe Config file
      ConfigParseOptions configParseOptions = ConfigUtils.getParseOptions();
      Config config = null;
      if (new File(args[0]).exists()) {
        try {
          config = ConfigFactory.parseFile(new File(args[0]).getCanonicalFile(), configParseOptions).resolve();
        } catch (IOException e) {
          System.err.println("Could not find config file: " + args[0]);
          System.exit(1);
        }
      } else {
        try {
          config = ConfigFactory.parseReader(IOUtils.getBufferedReaderFromClasspathOrFileSystem(args[0]), configParseOptions).resolve();
        } catch (IOException e) {
          System.err.println("Could not find config file: " + args[0]);
          System.exit(1);
        }
      }
      final Properties props = new Properties();
      for (Map.Entry<String, ConfigValue> entry : config.entrySet()) {
        String candidate = entry.getValue().unwrapped().toString();
        if (candidate != null) {
          props.setProperty(entry.getKey(), candidate);
        }
      }
      exec(() -> {
        Props.initializeAndValidate();
        toRun.apply(props);
      }, config);
    } else {
      // Case: Run with Properties file or command line arguments
      final Properties props = StringUtils.argsToProperties(args);
      Execution.exec(() -> {
        Props.initializeAndValidate();
        toRun.apply(props);
      }, props);
    }
  }

//  /** A shorthand for {@link edu.stanford.nlp.kbp.slotfilling.SlotfillingSystem#exec(java.util.function.Function, String[])} */
//  public static void exec(Consumer<Properties> fn, String[] args) {
//    exec((Properties in) -> { fn.accept(in); return null; }, args);
//  }

  /**
   * This is the main entry point for the KBP Slotfilling task.
   * @param args The command line arguments. @see exec
   */
  public static void main(String[] args) {
    exec((Properties props) -> {
      boolean train = Props.KBP_TRAIN;
      boolean evaluate = Props.KBP_EVALUATE;
      RunMode mode = (train && evaluate) ? RunMode.TRAIN_AND_EVALUATE : (train ? RunMode.TRAIN_ONLY : (evaluate ? RunMode.EVALUATE_ONLY : RunMode.CONSOLE));
      if (Props.KBP_VALIDATE) { mode = RunMode.VALIDATE; }
      try {
        runProgram(mode, props);
      } catch (Exception e) {
        logger.fatal(e);
      }
      logger.log(BLUE, "work dir: " + Props.WORK_DIR);
      return null;
    }, args);
  }
}
