package edu.stanford.nlp.kbp.common;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.util.*;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.entitylinking.EntityLinker;
import edu.stanford.nlp.kbp.entitylinking.TrainedEntityLinker;
import edu.stanford.nlp.kbp.slotfilling.classify.EnsembleRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.JointBayesRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.ModelType;
import edu.stanford.nlp.kbp.slotfilling.classify.OpenIERelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPEvaluator;
import edu.stanford.nlp.kbp.slotfilling.evaluate.GraphConsistencyPostProcessors.MergeStrategy;
import edu.stanford.nlp.kbp.slotfilling.evaluate.inference.BayesNetBuilder;
import edu.stanford.nlp.kbp.slotfilling.evaluate.inference.GraphInferenceEngine;
import edu.stanford.nlp.kbp.slotfilling.evaluate.inference.ProbabilisticGraphInferenceEngine;
import edu.stanford.nlp.kbp.slotfilling.evaluate.inference.SimpleGraphInferenceEngine;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.StandardIR;
import edu.stanford.nlp.kbp.slotfilling.ir.query.LuceneDocumentReader;
import edu.stanford.nlp.kbp.slotfilling.process.RelationFilter;
import edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer;
import edu.stanford.nlp.sequences.SeqClassifierFlags;
import edu.stanford.nlp.util.Execution.Option;
import edu.stanford.nlp.util.MetaClass;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

public class Props {
  /**

   * Some places in KBP temporarily change a global property.
   * To preserve thread safety, any time such those properties are changed
   * they should take this lock.
   */
  public static final Object PROPERTY_CHANGE_LOCK = "I'm a lock :)";

  //
  // KBP
  //
  @Option(name="work.dir", gloss="The directory to store logs, etc. in")
  public static File WORK_DIR = null;

  @Option(name="kbp.train", gloss="If set to true, train a KBP model")
  public static boolean KBP_TRAIN = false;
  @Option(name="kbp.evaluate", gloss="If set to true, evaluate the KBP model")
  public static boolean KBP_EVALUATE = false;
  @Option(name="kbp.validate", gloss="If set to true, validate slot fills")
  public static boolean KBP_VALIDATE = false;

  public static enum YEAR { KBP2009, KBP2010, KBP2011, KBP2012, KBP2013, KBP2014 }
  @Option(name="kbp.year", gloss="If true, logging will be more verbose")
  public static YEAR KBP_YEAR = YEAR.KBP2010;
  @Option(name="kbp.runid", gloss="The id of the run, for official evaluation")
  public static String KBP_RUNID = "stanford1";

  @Option(name="kbp.model.dir", gloss="The directory to save and load the KBP models from", required=true)
  public static File KBP_MODEL_DIR = new File("edu/stanford/nlp/kbp");
  @Option(name="kbp.verbose", gloss="If true, logging will be more verbose")
  public static boolean KBP_VERBOSE = false;
  @Option(name="kbp.email", gloss="A list of email addresses to send result statistics to")
  public static String[] KBP_EMAIL = new String[0];
  @Option(name="kbp.description", gloss="A very short description of this run")
  public static String KBP_DESCRIPTION = "";

  /** The model path; constructed on initialization */
  public static String KBP_MODEL_PATH;

  //
  // ENTITY LINKING
  //
  @Option(name="entitylinking.linker", gloss="The class of the entity linker to use for disambiguating entities")
  public static Class<? extends EntityLinker> KBP_ENTITYLINKER_CLASS = EntityLinker.GaborsHackyBaseline.class;
  /** The entity linker instance; constructed on initialization */
  public static Lazy<EntityLinker> ENTITYLINKING_LINKER = null;
  @Option(name="entitylinking.malenames", gloss="A list of male names")
  public static File ENTITYLINKING_MALENAMES = new File("edu/stanford/nlp/kbp/kbp_male_names.txt");
  @Option(name="entitylinking.femalenames", gloss="A list of female names")
  public static File ENTITYLINKING_FEMALENAMES = new File("edu/stanford/nlp/kbp/kbp_female_names.txt");
  @Option(name="entitylinking.abbreviations", gloss="A list of abbreviations")
  public static File ENTITYLINKING_ABBREVIATIONS = new File("edu/stanford/nlp/kbp/abbreviations.txt");
  @Option(name="entitylinking.entitylist", gloss="File path to store the entity list cache")
  public static final String ENTITYLINKING_ENTITY_LIST = "/scr/nlp/data/tackbp2014/data/Entities.gz";
  @Option(name="entitylinking.luceneindexdir", gloss="Lucene index directory for wiki data")
  public static final String ENTITYLINKING_LUCENE_INDEX_DIR = "/u/nlp/data/Wikipedia/wikitags/wikispans/solr/data/index";
  @Option(name="cache.entities.slurp", gloss="Load entities by slurping the database")
  public static boolean CACHE_ENTITIES_SLURP = false;
  @Option(name="entitylinking.numpositive", gloss="Number of positive examples to use")
  public static int ENTITYLINKING_NUMPOSITIVE = 5000;
  @Option(name="entitylinking.negativetopositiveratio", gloss="Ratio of negative to positive examples")
  public static double ENTITYLINKING_NEGATIVETOPOSITIVERATIO = 1.0;
  @Option(name="entitylinking.positivesameweight", gloss="Weight of positive same examples")
  public static float ENTITYLINKING_POSITIVE_SAME_WEIGHT = 1.0f;
  @Option(name="entitylinking.positivediffweight", gloss="Weight of positive different examples")
  public static float ENTITYLINKING_POSITIVE_DIFF_WEIGHT = 1.0f;
  @Option(name="entitylinking.negativesameweight", gloss="Weight of negative same examples")
  public static float ENTITYLINKING_NEGATIVE_SAME_WEIGHT = 1.0f;
  @Option(name="entitylinking.negativeapproxweight", gloss="Weight of negative approx examples")
  public static float ENTITYLINKING_NEGATIVE_APPROX_WEIGHT = 1.0f;
  @Option(name="entitylinking.negativediffweight", gloss="Weight of negative different examples")
  public static float ENTITYLINKING_NEGATIVE_DIFF_WEIGHT = 1.0f;
  @Option(name="entitylinking.personupweight", gloss="Up Weight for person examples")
  public static float ENTITYLINKING_PERSON_UP_WEIGHT = 1.0f;
  @Option(name="entitylinking.middlenameupweight", gloss="Up Weight of middle name mismatches")
  public static float ENTITYLINKING_MIDDLENAME_UP_WEIGHT = 1.0f;
  @Option(name="entitylinking.testfraction", gloss="Fraction of dataset that is test")
  public static double ENTITYLINKING_TESTFRACTION = 0.25;
  @Option(name="entitylinking.pathtolinker", gloss="Path to serialize linker")
  public static File ENTITYLINKING_PATHTOLINKER = new File("/scr/nlp/data/tackbp2014/models/models_entitylinking/best/linker.model");
  @Option(name="entitylinking.pathtolinkerjunit", gloss="Path to serialize junit linker")
  public static File ENTITYLINKING_PATHTOLINKERJUNIT = new File("/scr/nlp/data/tackbp2014/models/models_entitylinking/junit/linker.model");
  @Option(name="entitylinking.model", gloss="The model to run")
  public static String ENTITYLINKING_MODEL = "baseline";
  @Option(name="entitylinking.wikidictbackoff", gloss="The backoff linker to use")
  public static Class<? extends EntityLinker> ENTITYLINKING_WIKIBACKOFF_CLASS = EntityLinker.GaborsHackyBaseline.class;
  public static Lazy<EntityLinker> ENTITYLINKING_WIKIBACKOFF = null;


  //
  // PROCESS
  //
  @Option(name="process.regexner.dir", gloss="The directory with the RegexNER mappings")
  public static File PROCESS_REGEXNER_DIR = new File("edu/stanford/nlp/kbp");
  @Option(name="process.regexner.caseless", gloss="The file within process.regexner.dir which defines the caseless mapping")
  public static String PROCESS_REGEXNER_CASELESS = "kbp_regexner_mapping_nocase.tab";
  @Option(name="process.regexner.withcase", gloss="The file within process.regexner.dir which defines the cased mapping")
  public static String PROCESS_REGEXNER_WITHCASE = "kbp_regexner_mapping.tab";
  @Option(name="process.relation.normalizecorefslot", gloss="Get the normalized slot value from coref, rather than from the literal span")
  public static boolean PROCESS_RELATION_NORMALIZECOREFSLOT = false;
  @Option(name="process.wordclusters.file", gloss="File with mapping from words to clusters (tab se parated)")
  public static File PROCESS_WORDCLUSTERS_FILE = new File("edu/stanford/nlp/kbp/egw4-reut.512.clusters");
  @Option(name="process.newfeaturizer", gloss="Use the new (smaller) featurizer")
  public static boolean PROCESS_NEWFEATURIZER = false;

  @Option(name="process.domreader.countries")
  public static String PROCESS_DOMREADER_COUNTRIES = "edu/stanford/nlp/kbp/countries";
  @Option(name="process.domreader.manual.lists")
  public static String PROCESS_DOMREADER_MANUAL_LISTS = "edu/stanford/nlp/kbp/specific_relations";
  @Option(name="process.domreader.states")
  public static String PROCESS_DOMREADER_STATES = "edu/stanford/nlp/kbp/statesandprovinces";
  // TODO(gabor) See common.RelationType -- I've hard coded these in the code (they're not changing anytime soon)
  @Option(name="process.domreader.ner")
  public static File PROCESS_DOMREADER_NER = new File("edu/stanford/nlp/kbp/NER_types");


  //
  // INDEX
  //

  @Option(name="index.defaultversion", gloss="The version of the index -- this is generally going to be the year (e.g., 2011, 2013)")
  public static LuceneDocumentReader.KBPIndexVersion INDEX_DEFAULTVERSION = LuceneDocumentReader.KBPIndexVersion.KBP_INDEX_2013;
  @Option(name="index.paths", gloss="The paths to the KBP indices", required=true)
  public static File[] INDEX_PATHS = new File[0];
  public static enum QueryMode { NOOP, DUMB, REGULAR, BACKOFF, HEURISTIC_BACKOFF }
  @Option(name="index.mode", gloss="The type of querier to use: Regular does not back off, Backoff is a simple backoff, and HeuristicBackoff is a better backoff")
  public static QueryMode INDEX_MODE = QueryMode.NOOP;
  @Option(name="index.class", gloss="The class to use for implementing the IR component")
  public static Class<? extends KBPIR> INDEX_CLASS = StandardIR.class;
  @Option(name="index.test.sentences.per.entity", gloss="Number of sentences to use to find sentences for each entity" )
  public static int TEST_SENTENCES_PER_ENTITY = 50;
  @Option(name="index.official", gloss="The path of the official index", required=true)
  public static File INDEX_OFFICIAL = null;
  @Option(name="index.termindexdivisor", gloss="Lucene will take X times less memory for X times more runtime when searching the term dict")
  public static int INDEX_TERMINDEXDIVISOR = 1;  // see http://blog.mikemccandless.com/2010/07/lucenes-ram-usage-for-searching.html
  @Option(name="index.fastbackoff", gloss="If true, backoff to less entries -- this is likely faster, but less efficient")
  public static boolean INDEX_FASTBACKOFF = false;
  @Option(name="index.relationtriggers", gloss="File of keywords for each relation")
  public static File INDEX_RELATIONTRIGGERS = new File("edu/stanford/nlp/kbp/keywords_no_ml");
  @Option(name="index.reannotate", gloss="Annotators to use for re-annotating documents retrieved from the index (would normally be empty, but can be used to fix up annotations)")
  public static String INDEX_REANNOTATE = null;
  @Option(name="index.wikidict", gloss="Location of the Lucene index for the wikidict entity linking mappings")
  public static File INDEX_WIKIDICT = new File("/dev/null");
  @Option(name="index.readdoc.rewrite", gloss="A set of find/replace rules to apply to a serialized document file path")
  public static Map<String, String> INDEX_READDOC_REWRITE = new HashMap<>();

  @Option(name="index.coref.do", gloss="Search for and return sentences with coreferent entities.")
  public static boolean INDEX_COREF_DO = true;
  @Option(name="index.postirannotator.do", gloss="Use the new PostIRAnnotator when possible")
  public static boolean INDEX_POSTIRANNOTATOR_DO = true;
  @Option(name="index.postirannotator.approxname", gloss="Do approximate name matching on a first or last name if no full name exists in the article")
  public static boolean INDEX_POSTIRANNOTATOR_APPROXNAME = false;
  @Option(name="index.postirannotator.minlinkpercent", gloss="At least this percentage of links must be linked to the query entity for the query entity to be considered representative")
  public static double INDEX_POSTIRANNOTATOR_MINLINKPERCENT = 0.0;
  @Option(name="index.postirannotator.commonnames", gloss="Do approximate name matching on a first or last name if no full name exists in the article")
  public static File INDEX_POSTIRANNOTATOR_COMMONNAMES = new File("edu/stanford/nlp/kbp/common_names.txt");

  @Option(name="index.websnippets.do", gloss="If true, query for web sentences as well")
  public static boolean INDEX_WEBSNIPPETS_DO = false;
  @Option(name="index.websnippets.dir", gloss="The directory with annotated webqueries")
  public static File INDEX_WEBSNIPPETS_DIR = new File("/scr/nlp/data/tac-kbp/common-data/annotated_docs/websnippets");

  @Option(name="index.lucene.timeoutms", gloss="Lucene query timeout, in miliseconds. Avoid setting too big (or, set to Integer.MAX_VALUE outright)")
  public static int INDEX_LUCENE_TIMEOUTMS = Integer.MAX_VALUE;
  @Option(name="index.lucene.skippingbackoff", gloss="Skip documents if no results are found early. This is a useful tweak for speeding up datum caching")
  public static int INDEX_LUCENE_SKIPPINGBACKOFF = 0;
  @Option(name="index.lucene.abbreviations.do", gloss="If true, construct wildcard queries for abbreviations (this will be more accurate, but much slower)")
  public static boolean INDEX_LUCENE_ABBREVIATIONS_DO = false;

  @Option(name="index.train.sentences.per.entity", gloss="Skip documents if no results are found early. This is a useful tweak for speeding up datum caching")
  public static int TRAIN_SENTENCES_PER_ENTITY = 50;


  //
  // TRAIN
  //


  @Option(name="train.tuples.count", gloss="Number of KB training examples to train on")
  public static int TRAIN_TUPLES_COUNT = Integer.MAX_VALUE;
  @Option(name="train.tuples.files", gloss="Tuples stored in TSV files")
  public static String[] TRAIN_TUPLES_FILES = new String[]{};
  @Option(name="train.tuples.aux", gloss="Additional tuples stored in TSV files")
  public static String[] TRAIN_TUPLES_AUX = new String[]{};

  @Option(name="train.model", gloss="Model to train from")
  public static ModelType TRAIN_MODEL = ModelType.LR_INC;
  @Option(name="train.tuples.featurecountthreshold", gloss="Threshold for the minimum number of times a feature should occur")
  public static int FEATURE_COUNT_THRESHOLD = 5;
  @Option(name="train.spec.kbpmapping", gloss="Directory for mapping between input and KBP slots")
  public static File TRAIN_SPEC_KBP_MAPPING = new File("edu/stanford/nlp/kbp/slot_mapping");
  @Option(name="train.features", gloss="Features to use at training time", required=true)
  public static String[] TRAIN_FEATURES = new String[]{};
  @Option(name="train.dumpdataset", gloss="If true, print the training data to the debug channel. This can be very slow")
  public static boolean TRAIN_DUMPDATASET = false;
  @Option(name="train.fixkb", gloss="Fix the knowledge base, enforcing type consistency of relations when passed into training")
  public static boolean TRAIN_FIXKB = false;

  @Option(name="train.negatives.ratio", gloss="Subsample negative examples by this ratio")
  public static double TRAIN_NEGATIVES_RATIO = 0.5;
  @Option(name="train.negatives.incomplete", gloss="Only treat slot fills as negative if we know the correct slot fill")
  public static boolean TRAIN_NEGATIVES_INCOMPLETE = true;
  @Option(name="train.negatives.incompatible", gloss="If train.negatives.incomplete is enabled, also add incompatble relations")
  public static boolean TRAIN_NEGATIVES_INCOMPATIBLE = true;
  @Option(name="train.negatives.singetons", gloss="If true, try to only train on negatives which have a single sentence justification")
  public static boolean TRAIN_NEGATIVES_SINGLETONS = false;

  @Option(name="train.unlabeled.do", gloss="If train.unlabeled is enabled, also include unlabeled datums for which we will attempt to guess if they are positive or negative")
  public static boolean TRAIN_UNLABELED = false;
  @Option(name="train.unlabeled.select", gloss="Method to use for selecting unlabeled examples.")
  public static KBPTrainer.UnlabeledSelectMode TRAIN_UNLABELED_SELECT = KBPTrainer.UnlabeledSelectMode.NOT_LABELED;


  @Option(name="train.annotatedsentences.do", gloss="If true, use annotated sentences")
  public static boolean TRAIN_ANNOTATED_SENTENCES_DO = false;
  @Option(name="train.annotatedsentences.data", gloss="A set of csv files to read in which have annotated labels for individual sentences (mentions).")
  public static File[] TRAIN_ANNOTATED_SENTENCES_DATA = new File[0];
  @Option(name="train.annotatedsentences.keys", gloss="An (optional) set of keys to read in as the annotated sentences to use.")
  public static File TRAIN_ANNOTATED_SENTENCES_KEYS = new File("");  // leave blank or null to disable

  @Option(name="train.tokensregex.dir", gloss="Directory for tokenregex rules")
  public static File TRAIN_TOKENSREGEX_DIR = new File("edu/stanford/nlp/kbp");
  
  @Option(name="train.semgrex.dir", gloss="Directory for semgrex rules")
  public static File TRAIN_SEMGREX_DIR = new File("edu/stanford/nlp/kbp/semgrexrules");
  
  @Option(name="train.lr.allnegatives", gloss="If true, use anything that's not a positive example as a negative example for regression models")
  public static boolean TRAIN_LR_ALLNEGATIVES = false;

  @Option(name="train.supervised.reweight", gloss="If true, rew-weight datums match the empirical distribution of labels in the dataset")
  public static boolean TRAIN_SUPERVISED_REWEIGHT = false;
  @Option(name="train.supervised.threshold", gloss="The threshold above which to propose a relation as true")
  public static double TRAIN_SUPERVISED_THRESHOLD = 0.5;

  @Option(name="train.jointbayes.zminimizer", gloss="Minimizer to use for training z classifier")
  public static KBPTrainer.MinimizerType TRAIN_JOINTBAYES_ZMINIMIZER = KBPTrainer.MinimizerType.QN; // QN gives best results, use SGD for faster training
  @Option(name="train.jointbayes.percent.positive", gloss="The percent of bags (entity, slot value, relation) that should be positive.")
  public static double TRAIN_JOINTBAYES_PERCENT_POSITIVE = 0.25;
  @Option(name="train.jointbayes.filter", gloss="The threshold filter to place on the Z and Y labels")
  public static Class<? extends JointBayesRelationExtractor.LocalFilter> TRAIN_JOINTBAYES_FILTER = JointBayesRelationExtractor.AllFilter.class;
  public static enum TRAIN_JOINTBAYES_OUTDISTRIBUTION_TYPES {Y_GIVEN_ZSTAR, NOISY_OR, Y_THEN_NOISY_OR}
  @Option(name="train.jointbayes.outdistribution", gloss="Ouput a probability, or take score compared to UNRELATED tag")
  public static TRAIN_JOINTBAYES_OUTDISTRIBUTION_TYPES TRAIN_JOINTBAYES_OUTDISTRIBUTION = TRAIN_JOINTBAYES_OUTDISTRIBUTION_TYPES.Y_THEN_NOISY_OR;
  @Option(name="train.jointbayes.folds", gloss="The number of folds to train the Joint Bayes relation extractor on")
  public static int TRAIN_JOINTBAYES_FOLDS = 3;
  @Option(name="train.jointbayes.loadinitmodel", gloss="Allows the initial model to be re-read from disk, for efficiency")
  public static boolean TRAIN_JOINTBAYES_LOADINITMODEL = false;
  @Option(name="train.jointbayes.initialization", gloss="MIML-RE Initialization -- one of distsup or supervised")
  public static JointBayesRelationExtractor.Initialization TRAIN_JOINTBAYES_INITIALIZATION = JointBayesRelationExtractor.Initialization.DISTSUP;
  @Option(name="train.jointbayes.turkertrust", gloss="The degree of trust placed in turker annotations -- between 0 (disable annotations) and 1 (hard assignment)")
  public static double TRAIN_JOINTBAYES_TURKERTRUST = 1.0;
  @Option(name="train.jointbayes.loadinitmodel.file", gloss="Specifies which initial model to be read (for testing)")
  public static String TRAIN_JOINTBAYES_LOADINITMODEL_FILE = null;
  public static enum Y_FEATURE_CLASS { ATLEAST_ONCE, COOC, UNIQUE, ATLEAST_N, SIGMOID}
  @Option(name="train.jointbayes.yfeatures", gloss="Set of features to use for the y classifier")
  private static Set<String> TRAIN_JOINTBAYES_YFEATURES_INTERNAL = new HashSet<>(Arrays.<String>asList(Y_FEATURE_CLASS.ATLEAST_ONCE.name(), Y_FEATURE_CLASS.COOC.name()));
  public static Set<Y_FEATURE_CLASS> TRAIN_JOINTBAYES_YFEATURES = new HashSet<>(); // Use me externally
  @Option(name="train.jointbayes.zsigma", gloss="Regularization param for zClassifier.  Low sigma => more regularization.")
  public static double TRAIN_JOINTBAYES_ZSIGMA = 1.0;
  @Option(name="train.jointbayes.epochs", gloss="The number of epochs to run JointBayes EM for")
  public static int TRAIN_JOINTBAYES_EPOCHS = 5;
  @Option(name="train.jointbayes.inferencetype", gloss="The type of Y inference to do (e.g., \"stable\")")
  public static JointBayesRelationExtractor.InferenceType TRAIN_JOINTBAYES_INFERENCETYPE = JointBayesRelationExtractor.InferenceType.STABLE;
  @Option(name="train.jointbayes.trainy")
  public static boolean TRAIN_JOINTBAYES_TRAINY = true;
  @Option(name="train.jointbayes.multithread", gloss="If set to false, MIML-RE will not multithread.")
  public static boolean TRAIN_JOINTBAYES_MULTITHREAD = true;

  @Option(name="train.ensemble.method", gloss="The type of model combination to use (e.g., Bagging or Sub-Bagging")
  public static EnsembleRelationExtractor.EnsembleMethod TRAIN_ENSEMBLE_METHOD = EnsembleRelationExtractor.EnsembleMethod.BAGGING;
  @Option(name="train.ensemble.component", gloss="The type of classifiers to train")
  public static ModelType TRAIN_ENSEMBLE_COMPONENT = ModelType.JOINT_BAYES;
  @Option(name="train.ensemble.numcomponents", gloss="The number of classifiers to train")
  public static int TRAIN_ENSEMBLE_NUMCOMPONENTS = 7;
  @Option(name="train.ensemble.bagsize", gloss="The size of each bag")
  public static int TRAIN_ENSEMBLE_BAGSIZE = TRAIN_TUPLES_COUNT;
  @Option(name="train.ensemble.subagredundancy", gloss="The number of components to add a datum too (e.g., 3 means a datum appears in 3 classifiers' training sets")
  public static int TRAIN_ENSEMBLE_SUBAGREDUNDANCY = 1;

  @Option(name="train.perceptron.epochs", gloss="The number of epochs to train the Perceptron extractor for")
  public static int PERCEPTRON_EPOCHS = 20;
  @Option(name="train.perceptron.normalize")
  public static String PERCEPTRON_NORMALIZE = "L2J";
  @Option(name="train.perceptron.softmax")
  public static boolean PERCEPTRON_SOFTMAX = true;


  //
  // TEST
  //
  @Option(name="test.auxmodels", gloss="Additional models to use during testing")
  public static ModelType[] TEST_AUXMODELS = new ModelType[0];

  public static File TEST_QUERIES = null;  // filled in at initialization
  @Option(name="test.queries", gloss="The location of the test queries to evaluate on")
  private static Map<String, String> TEST_QUERIES_BY_YEAR = new HashMap<String, String>() {{
    put(YEAR.KBP2009.name(), "edu/stanford/nlp/kbp/evaluation_queries/2009.xml");
    put(YEAR.KBP2010.name(), "edu/stanford/nlp/kbp/evaluation_queries/2010.xml");
    put(YEAR.KBP2011.name(), "edu/stanford/nlp/kbp/evaluation_queries/2011.xml");
    put(YEAR.KBP2012.name(), "edu/stanford/nlp/kbp/evaluation_queries/2012.xml");
    put(YEAR.KBP2013.name(), "edu/stanford/nlp/kbp/evaluation_queries/2013.xml");
    put(YEAR.KBP2014.name(), "edu/stanford/nlp/kbp/evaluation_queries/2014.xml");
  }};
  public static Maybe<File> TEST_RESPONSES = null;  // filled in at initialization
  @Option(name="test.responses", gloss="The location of the gold responses to the test queries")
  private static Map<String, String> TEST_RESPONSES_BY_YEAR = new HashMap<String, String>() {{
    put(YEAR.KBP2009.name(), "edu/stanford/nlp/kbp/evaluation_results/2009.tab");
    put(YEAR.KBP2010.name(), "edu/stanford/nlp/kbp/evaluation_results/2010.tab");
    put(YEAR.KBP2011.name(), "edu/stanford/nlp/kbp/evaluation_results/2011.dir");
    put(YEAR.KBP2012.name(), "edu/stanford/nlp/kbp/evaluation_results/2012.dir");
    put(YEAR.KBP2013.name(), "edu/stanford/nlp/kbp/evaluation_results/2013.dir");
  }};

  @Option(name="test.consistency.do", gloss="If false, don't run the consistency component")
  public static boolean TEST_CONSISTENCY_DO = true;
  @Option(name="test.consistency.worldknowledgedir", gloss="The directory where the world knowledge files live")
  public static File TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR = new File("edu/stanford/nlp/kbp");
  public static enum GibbsObjective{ TOP, SUM }
  @Option(name="test.consistency.gibbsobjective", gloss="The objective for the Gibbs sampler to optimize")
  public static GibbsObjective TEST_CONSISTENCY_GIBBSOBJECTIVE = GibbsObjective.TOP;
  @Option(name="test.consistency.mixingtime", gloss="The mixing time for gibbs sampling")
  public static int TEST_CONSISTENCY_MIXINGTIME = 1000;
  @Option(name="test.consistency.rewrite", gloss="If true, allow post processors to rewrite a slot value")
  public static boolean TEST_CONSISTENCY_REWRITE = false;
  @Option(name="test.consistency.flexibletypes", gloss="If true, allow for some org relations to be translated to their per equivalients and visa versa")
  public static boolean TEST_CONSISTENCY_FLEXIBLETYPES = false;

  @Option(name="test.relationextractor.alternatenames.do", gloss="If false, do not propose alternate names from the relation extractor")
  public static boolean TEST_RELATIONEXTRACTOR_ALTERNATENAMES_DO = true;

  @Option(name="test.graph.depth", gloss="Depth of the entity graph" )
  public static int TEST_GRAPH_DEPTH = 4;
  @Option(name="test.graph.maxsize", gloss="Upper limit on the number of vertices in a graph" )
  public static int TEST_GRAPH_MAXSIZE = 150;
  @Option(name="test.graph.merge.do", gloss="Merge equivalent entites?" )
  public static boolean TEST_GRAPH_MERGE_DO = false;
  @Option(name="test.graph.merge.strategy", gloss="Strategy to merge the edges between 'equivalent slots'" )
  public static MergeStrategy TEST_GRAPH_MERGE_STRATEGY = MergeStrategy.NOISY_OR;
  @Option(name="test.graph.symmeterize.do", gloss="Do graph symmeterization" )
  public static boolean TEST_GRAPH_SYMMETERIZE_DO = false;
  @Option(name="test.graph.transitive.do", gloss="Compute transitive completion of relations" )
  public static boolean TEST_GRAPH_TRANSITIVE_DO = false;
  @Option(name="test.graph.openie.do", gloss="Use ReVerb extractions" )
  public static boolean TEST_GRAPH_OPENIE_DO = false;
  @Option(name="test.graph.openie.class", gloss="The OpenIE relation extractor to load" )
  public static Class<? extends OpenIERelationExtractor> TEST_GRAPH_OPENIE_CLASS = OpenIERelationExtractor.class;
  @Option(name="test.graph.openie.prune", gloss="If true, prune the ReVerb graph to contain only known useful relations" )
  public static boolean TEST_GRAPH_OPENIE_PRUNE = false;
  @Option(name="test.graph.kbp.do", gloss="Use KBP extractions; defaults to 'true' for obvious reasons." )
  public static boolean TEST_GRAPH_KBP_DO = true;
  @Option(name="test.graph.inference.do", gloss="Run graph inference procedures?" )
  public static boolean TEST_GRAPH_INFERENCE_DO = false;
  @Option(name="test.graph.inference.class", gloss="The type of Graph Inference Engine to use" )
  public static Class<? extends GraphInferenceEngine> TEST_GRAPH_INFERENCE_CLASS = SimpleGraphInferenceEngine.class;
  @Option(name="test.graph.inference.rules", gloss="An (ordered) comma separated list of files containing weighted inference rules" )
  public static String TEST_GRAPH_INFERENCE_RULES = "/scr/nlp/data/tackbp2014/data/inference_rules/kbp.prior.rules, /scr/nlp/data/tackbp2014/data/inference_rules/kbp.v4.3.minweight5.smoothAuto.rules" ;
  public static List<File> TEST_GRAPH_INFERENCE_RULES_FILES;
  @Option(name="test.graph.inference.rules.cutoff", gloss="Confidence threshold when reading rules" )
  public static double TEST_GRAPH_INFERENCE_RULES_CUTOFF = 0.0;
  @Option(name="test.graph.inference.rules.mode", gloss="How big can the antecedent be?" )
  public static ProbabilisticGraphInferenceEngine.RulesMode TEST_GRAPH_INFERENCE_RULES_MODE = ProbabilisticGraphInferenceEngine.RulesMode.REVERB_STRICT;
  @Option(name="test.graph.inference.rules.restrictNonQuery", gloss="Assume that any absent edge between non-query entities is evidence of its falsehood")
  public static boolean TEST_GRAPH_INFERENCE_RULES_RESTRICT_NON_QUERY = false;
  @Option(name="test.graph.inference.doMAP", gloss="Use MAP inference (instead of marginal inference)")
  public static boolean TEST_GRAPH_INFERENCE_DO_MAP = false;
  @Option(name="test.graph.inference.rule.merge.method", gloss="Method to combine two rules")
  public static BayesNetBuilder.FactorMergeMethod TEST_GRAPH_INFERENCE_RULE_MERGE_METHOD = BayesNetBuilder.FactorMergeMethod.GEOMETRIC_MEAN;
  @Option(name="test.graph.inference.gibbs.samples", gloss="Number of samples to use with Gibbs")
  public static int TEST_GRAPH_INFERENCE_GIBBS_SAMPLES = (int) 6e6;
  @Option(name="test.graph.inference.prior", gloss="Prior to use")
  public static double TEST_GRAPH_INFERENCE_PRIOR = 0.3;


  @Option(name="test.graph.inference.hacks.alwaysTrue", gloss="Hack to set priors always true")
  public static boolean TEST_GRAPH_INFERENCE_HACKS_ALWAYS_TRUE = false;
  @Option(name="test.graph.inference.hacks.noSpouse", gloss="Hack to ignore rules inferring spouses")
  public static boolean TEST_GRAPH_INFERENCE_HACKS_NO_SPOUSE = false;
  @Option(name="test.graph.inference.hacks.noNegativeTranslations", gloss="Remove all negative weighted translation rules")
  public static boolean TEST_GRAPH_INFERENCE_HACKS_NO_NEGATIVE_TRANSLATIONS = false;
  @Option(name="test.graph.inference.hacks.softPriors", gloss="Priors between 0.5 and 0.9 instead of 0.5 and 1.0")
  public static boolean TEST_GRAPH_INFERENCE_HACKS_SOFT_PRIORS = false;
  @Option(name="test.graph.inference.hacks.globalConsistency", gloss="Do global consistency before running.")
  public static boolean TEST_GRAPH_INFERENCE_HACKS_GLOBAL_CONSISTENCY = true;

  @Option(name="test.graph.altnames.do", gloss="If true, propose alternate names when merging entities" )
  public static boolean TEST_GRAPH_ALTNAMES_DO = false;

  @Option(name="test.graph.inference.depth", gloss="Depth of inference rules to run inference over. E.g., 3 means A->B->C is valid, and A->B->C->A is valid, but not A->B->C->D.")
  public static int TEST_GRAPH_INFERENCE_DEPTH = 3;

  @Option(name="test.graph.document.budget", gloss="Entity budget limitting the number of entities added per document" )
  public static int TEST_GRAPH_DOCUMENT_BUDGET = 1000;

  @Option(name="test.sentences.max.tokens", gloss="Largest number of tokens to consider in a given sentence" )
  public static int TEST_SENTENCES_MAX_TOKENS = 100;
  @Option(name="test.relationfilter.do", gloss="Do within-sentence relation filtering")
  public static boolean TEST_RELATIONFILTER_DO = false;
  @Option(name="test.relationfilter.components", gloss="Comma delimmited list of filter components")
  public static Class[] TEST_RELATIONFILTER_COMPONENTS = new Class[]{
      RelationFilter.CorefFilterComponent.class,
      RelationFilter.PerRelTypeCompetitionFilterComponent.class,
    };
  @Option(name="test.slotfilling.mode", gloss="Can be either 'simple' (default) or 'inferential'; the latter uses cross-document relational inference.")
  public static KBPEvaluator.SlotFillerType TEST_SLOTFILLING_MODE = KBPEvaluator.SlotFillerType.SIMPLE;
  @Option(name="test.goldir", gloss="If true, the IR component will dutifully return the gold documents for a query")
  public static boolean TEST_GOLDIR = false;
  @Option(name="test.goldslots", gloss="If true, simply return all the correct answers")
  public static boolean TEST_GOLDSLOTS = false;
  @Option(name="test.probabilitypriors", gloss="If true, use Bayes Rule to convert P(r | e1, e2) to P(e2 | e1)")
  public static boolean TEST_PROBABILITYPRIORS = false;
  @Option(name="test.provenance.do", gloss="If set to false, ignore provenance altogether")
  public static boolean TEST_PROVENANCE_DO = true;
  @Option(name="test.provenance.recover", gloss="Try to find a provenance for the slot fill via IR, if not found. This is only for slot fills by non-rule-based-classifiers.")
  public static boolean TEST_PROVENANCE_RECOVER = false;
  @Option(name="test.provenance.recover.rules", gloss="Try to find a provenance for the slot fill via IR for rules, if not found.")
  public static boolean TEST_PROVENANCE_RECOVER_RULES = false;
  @Option(name="test.provenance.recover.nonclassify", gloss="works only if test.provenance.recover is true; if finding provenance via the classifiers fail then heurisitically select a provenance sentence.")
  public static boolean TEST_PROVENANCE_RECOVER_NONCLASSIFY = false;
  @Option(name="test.scoremode", gloss="Test on system retrieved recall, or the recall from all systems.")
  public static KBPEvaluator.ScoreMode TEST_SCORE_MODE = KBPEvaluator.ScoreMode.OFFICIAL;
  @Option(name="test.anydoc", gloss="If true, accept any document as provenance")
  public static boolean TEST_ANYDOC = true;
  @Option(name="test.rules.alternatenames.do", gloss="Use rule-based classifiers to augment the main classifier")
  public static boolean TEST_RULES_ALTERNATENAMES_DO = false;
  @Option(name="test.rules.alternatenames.fraction", gloss="The fraction of all mentions an alternate name has to occur to be considered valid (safeguards against bad IR)")
  public static double TEST_RULES_ALTERNATENAMES_FRACTION = 0.00;
  @Option(name="test.rules.alternatenames.coref", gloss="The fraction of all mentions an alternate name has to occur to be considered valid (safeguards against bad IR)")
  public static boolean TEST_RULES_ALTERNATENAMES_COREF = false;
  @Option(name="test.query.name", gloss="Restrict to this entity.")
  public static String TEST_QUERY_NAME = "";
  @Option(name="test.nqueries", gloss="Number of test queries. Set for debugging. Defaults to all the queries")
  public static int TEST_NQUERIES = Integer.MAX_VALUE;
  @Option(name="test.querystart", gloss="Index of first test query to process.  This property is ignored if test.nqueries is null.")
  public static int TEST_QUERYSTART = 0;
  
  public static enum TUNE_MODE {NONE, FIXED, GLOBAL, FIXED_PER_RELATION, PER_RELATION }
  @Option(name="test.threshold.tune", gloss="Tune the threshold for the minimum confidence for slots")
  public static TUNE_MODE TEST_THRESHOLD_TUNE = TUNE_MODE.FIXED;
  @Option(name="test.threshold.min.perrelation", gloss="Tune the threshold for accepted slots for every relation")
  public static Double[] TEST_THRESHOLD_MIN_PERRELATION = CollectionUtils.map(RelationType.values(), in -> 0.0);
  @Option(name="test.threshold.min.global", gloss="Set the threshold for the minimum confidence for slots")
  public static double TEST_THRESHOLD_MIN_GLOBAL = 0.5;
  @Option(name="test.threshold.jointbayes.default", gloss="The threshold above which to classify a default relation type as /true/")
  public static double TEST_THRESHOLD_JOINTBAYES_DEFAULT = 0.5;
  public static Map<String, Double> TEST_THRESHOLD_JOINTBAYES_PERRELATION = new HashMap<>();
  @SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
  @Option(name="test.threshold.jointbayes.perrelation", gloss="The threshold above which to classify a default relation type as /true/, tuned per relation")
  private static Map<String, String> TEST_THRESHOLD_JOINTBAYES_PERRELATION_IMPL = new HashMap<>();

  @Option(name="test.list.output")
  public static KBPEvaluator.ListOutput TEST_LIST_OUTPUT = KBPEvaluator.ListOutput.ALL;
  @Option(name="test.queryscorefile")
  public static String TEST_QUERYSCOREFILE = "query_score";

  public static enum EnsembleCombinationMethod {AGREE_ANY, AGREE_ALL, AGREE_MOST, AGREE_TWO, AGREE_FIRST}
  @Option(name="test.ensemble.combination")
  public static EnsembleCombinationMethod TEST_ENSEMBLE_COMBINATION = EnsembleCombinationMethod.AGREE_MOST;

  //
  // DEEP DIVE
  //
  @Option(name="deepdive.host")
  public static String DEEPDIVE_HOST = System.getenv("PGHOST");
  @Option(name="deepdive.port")
  public static int DEEPDIVE_PORT = System.getenv("PGPORT") != null ? Integer.parseInt(System.getenv("PGPORT")) : 5432;
  @Option(name="deepdive.user")
  public static String DEEPDIVE_USERNAME = System.getenv("PGUSER");
  @Option(name="deepdive.password")
  public static String DEEPDIVE_PASSWORD = System.getenv("PGPASSWORD");
  @Option(name="deepdive.db", gloss="the database name to query DeepDive")
  public static String DEEPDIVE_DB = System.getenv("DBNAME");
  @Option(name="deepdive.threshold.instance", gloss="The minimum probability with which to mark a relation instance as true.")
  public static Object DEEPDIVE_THRESHOLD_INSTANCE = 0.5;
  @Option(name="deepdive.threshold.mention", gloss="The minimum probability with which to mark a relation mention as true.")
  public static Object DEEPDIVE_THRESHOLD_MENTION = 0.5;
  @Option(name="deepdive.threshold.link", gloss="The minimum probability with which to mark an entity link as true.")
  public static Object DEEPDIVE_THRESHOLD_LINK = 0.5;

  //
  // VALIDATE
  //
  public static Maybe<File> VALIDATION_QUERIES = Maybe.Nothing();  // filled in at initialization
  @Option(name="validate.queries", gloss="The location of the test queries to evaluate on")
  private static Map<String, String> VALIDATION_QUERIES_BY_YEAR = new HashMap<String, String>() {{
    put(YEAR.KBP2013.name(), "/scr/nlp/data/tackbp2013/data/2013/validation_inputs/");
  }};
  @Option(name="validate.forceclassifiable", gloss="Force any example marked 'valid' to have a sentence we could recognize and classify")
  public static boolean VALIDATE_FORCECLASSIFIABLE = false;
  @Option(name="validate.forcetype", gloss="Force any example marked 'valid' to have a recognized NER type")
  public static boolean VALIDATE_FORCETYPE = false;
  @Option(name="validate.filternorelation", gloss="Filter examples which are classified as 'no relation' but were not classified as another relation")
  public static boolean VALIDATE_FILTERNORELATION = false;
  @Option(name="validate.rules.do", gloss="Use rule-based classifiers to augment the main classifier")
  public static boolean VALIDATE_RULES_DO = false;
  @Option(name="validate.runslotfiller", gloss="If true, run our full slotfiller on any entity/")
  public static boolean VALIDATE_RUNSLOTFILLER = false;

  //
  // CACHE
  //
  @Option(name="cache.lock", gloss="If true, try to lock files whenever possible for caching")
  public static boolean CACHE_LOCK = false;
  @Option(name="cache.sentences.do", gloss="Cache directory for sentence IR extractions")
  public static boolean CACHE_SENTENCES_DO = false;
  @Option(name="cache.sentences.redo", gloss="Overwrite the sentence cache with a new IR retrieval")
  public static boolean CACHE_SENTENCES_REDO = false;
  @Option(name="cache.datums.do", gloss="Cache datums (IR + annotation)")
  public static boolean CACHE_DATUMS_DO = false;
  @Option(name="cache.datums.redo", gloss="Overwrite the datum cache with new datums")
  public static boolean CACHE_DATUMS_REDO = false;
  @Option(name="cache.datums.slurp", gloss="Load datums by slurping the database, rather than going through the seed queries")
  public static boolean CACHE_DATUMS_SLURP = false;
  @Option(name="cache.datums.ordered", gloss="Read the tuples from the cache in a fixed, deterministic order")
  public static boolean CACHE_DATUMS_ORDER = true;
  @Option(name="cache.provenance.do", gloss="Cache provenance of an entity pair")
  public static boolean CACHE_PROVENANCE_DO = false;
  @Option(name="cache.provenance.redo", gloss="Re-cache provenances")
  public static boolean CACHE_PROVENANCE_REDO = false;
  @Option(name="cache.sentencegloss.do", gloss="Cache sentence gloss of a datum")
  public static boolean CACHE_SENTENCEGLOSS_DO = false;
  @Option(name="cache.graph.do", gloss="Cache the raw extracted graphs for a given entity to fill slots for")
  public static boolean CACHE_GRAPH_DO = false;
  @Option(name="cache.graph.redo", gloss="Overwrite the graph cache with newly computed graphs")
  public static boolean CACHE_GRAPH_REDO = false;

  //
  // POSTGRES
  //
  @Option(name="psql.host", gloss="The hostname for the PSQL server")
  public static String PSQL_HOST = "example.com";  // for the NLP machines: jonsson
  @Option(name="psql.host.tunnels", gloss="A list of hosts which don't have direct access, but tunnel through localhost")
  public static String[] PSQL_HOST_TUNNELS = new String[]{"zouk".intern(), "roshini".intern()};
  @Option(name="psql.port", gloss="The port at which postgres is running")
  public static int PSQL_PORT = 5432;
  @Option(name="psql.db", gloss="The database name")
  public static String PSQL_DB = "kbp";
  @Option(name="psql.username", gloss="The username for the postgres session")
  public static String PSQL_USERNAME = "kbp";
  @Option(name="psql.password", gloss="The password for the postgres session")
  public static String PSQL_PASSWORD = "kbp";
  @Option(name="psql.batch", gloss="If true, batch writes to PSQL.")
  public static boolean PSQL_BATCH = true;

  @Option(name="psql.tuffy.db", gloss="The database to store Tuffy data in")
  public static String PSQL_TUFFY_DB = "tuffy_kbp";
  @Option(name="psql.tuffy.username", gloss="The username for Tuffy's postgres session")
  public static String PSQL_TUFFY_USERNAME = "tuffer";
  @Option(name="psql.tuffy.password", gloss="The password for Tuffy's postgres session")
  public static String PSQL_TUFFY_PASSWORD = "tuffer";

  //
  // SINGLE PASS
  //

  @Option(name="shallowdive.officialonly", gloss="If true, only slurp the official table")
  public static boolean SHALLOWDIVE_OFFICIALONLY = false;
  @Option(name="shallowdive.cachewikidict", gloss="If true, cache the Wikidict dictionary (~5GB; 5 minutes)")
  public static boolean SHALLOWDIVE_CACHEWIKIDICT = false;

  @Option(name="shallowdive.featurize.redo", gloss="If true, redo the featurization step")
  public static boolean SHALLOWDIVE_FEATURIZE_REDO = false;
  @Option(name="shallowdive.featurize.openie", gloss="If true, extract features from OpenIE")
  public static boolean SHALLOWDIVE_FEATURIZE_OPENIE = false;
  @Option(name="shallowdive.featurize.output", gloss="The table to output features to")
  public static String SHALLOWDIVE_FEATURIZE_OUTPUT = "shallowdive_datums";
  @Option(name="shallowdive.featurize.chunksize", gloss="The number of documents to parse per thread")
  public static int SHALLOWDIVE_FEATURIZE_CHUNKSIZE = 1000;

  @Option(name="shallowdive.train.redo", gloss="If true, redo the training step")
  public static boolean SHALLOWDIVE_TRAIN_REDO = false;
  @Option(name="shallowdive.train.datums", gloss="The table to read training datums from")
  public static String SHALLOWDIVE_TRAIN_DATUMS = "shallowdive_datums";

  @Option(name="shallowdive.evaluate.do", gloss="If true, actually run the evaluation script")
  public static boolean SHALLOWDIVE_EVALUATE_DO = false;
  @Option(name="shallowdive.evaluate.lazy", gloss="If true, don't require the knowledge base to be populated.")
  public static boolean SHALLOWDIVE_EVALUATE_LAZY = false;
  @Option(name="shallowdive.evaluate.kb", gloss="The knowledge base table to evaluate from")
  public static String SHALLOWDIVE_EVALUATE_KB = "shallowdive_kb";
  @Option(name="shallowdive.evaluate.datums", gloss="The knowledge base table to evaluate from")
  public static String SHALLOWDIVE_EVALUATE_DATUMS = "shallowdive_datums";


  //
  // MISC
  //
  @Option(name="junit", gloss="An option for whether we're in Junit (disables some checks)")
  public static boolean JUNIT = false;
  @Option(name="rdf.server.uri")
  public static String RDF_SERVER_URI = "rdf.server.uri";

  //
  // HACKS
  // None of these should be enabled by default
  //

  @Option(name="hacks.disallowduplicatedatums", gloss="Explicitly check for and remove duplicate datums at training and test time")
  public static boolean HACKS_DISALLOW_DUPLICATE_DATUMS = false;
  @Option(name="hacks.dontreadkb", gloss="Try to avoid reading the knowledge base as much as possible. This options should never be set in production.")
  public static boolean HACKS_DONTREADKB = false;
  @Option(name="hacks.squashrandom", gloss="Remove all chance of randomization (where I've found it at least), even if the randomization is deterministic")
  public static boolean HACKS_SQUASHRANDOM = false;
  @Option(name="hacks.oldindexserialization", gloss="The old index was serialized with different Kryo properties -- use those properties")
  public static boolean HACKS_OLDINDEXSERIALIZATION = false;
  @Option(name="hacks.hackymodelcombination", gloss="Proof of concept model combination (don't judge me!)")
  public static boolean HACKS_HACKYMODELCOMBINATION = false;
  @Option(name="hacks.forcebaselinelinkerinconsistency", gloss="If true, force my hacky baseline in the consistency stage")
  public static boolean HACKS_FORCEBASELINELINKERINCONSISTENCY = false;
  @Option(name="hacks.marker", gloss="An arbitrary marker. For hacks and stuff.")
  public static boolean HACKS_MARKER = false;

  //
  // INITIALIZATION
  //

  static {
    try {
      WORK_DIR = Files.createTempDirectory("kbp.workdir").toFile();
      WORK_DIR.deleteOnExit();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

  }

  public static void initializeAndValidate() {
    // Directories must exist
    if (!Props.KBP_MODEL_DIR.exists()) {
      if( !Props.KBP_MODEL_DIR.mkdirs() ) {
        throw new IllegalStateException("Could not create model directory: " + KBP_MODEL_DIR);
      }
    }
    // Create model path
    Props.KBP_MODEL_PATH = Props.KBP_MODEL_DIR.getPath() + File.separator + "kbp_relation_model." + Props.TRAIN_MODEL.name() + "." + (int) (100.0 * Props.TRAIN_NEGATIVES_RATIO) + Props.SER_EXT;
    // Error Checks
    if (!WORK_DIR.exists()) { if (!WORK_DIR.mkdirs()) { throw new IllegalStateException("Could not create work dir"); } }
    if (Props.TEST_THRESHOLD_MIN_PERRELATION.length != RelationType.values().length) {
      throw new IllegalStateException("per-slot thresholds (test.threshold.min.perrelation) have to be equal to the number of relations");
    }
    for (Class c : TEST_RELATIONFILTER_COMPONENTS) {
      if (c == null) { throw new IllegalStateException("Did not properly cast TEST_RELATIONFILTER_COMPONENTS"); }
      if (!RelationFilter.FilterComponent.class.isAssignableFrom(c)) {
        throw new IllegalStateException("Class is not a FilterComponent: " + c);
      }
    }
    // Initialize Queries
    TEST_QUERIES = new File(TEST_QUERIES_BY_YEAR.get(KBP_YEAR.name()));
    if (KBP_EVALUATE && !TEST_QUERIES.exists()) {
      throw new IllegalArgumentException("Could not find test queries: " + TEST_QUERIES);
    }
    if (TEST_RESPONSES_BY_YEAR.containsKey(KBP_YEAR.name())) {
      TEST_RESPONSES = Maybe.Just(new File(TEST_RESPONSES_BY_YEAR.get(KBP_YEAR.name())));
      if (KBP_EVALUATE && !TEST_RESPONSES.get().exists()) {
        throw new IllegalArgumentException("Could not find test responses: " + TEST_RESPONSES);
      }
      // Copy test responses to work dir
      for (File keyFile : TEST_RESPONSES) {
        if (keyFile.isFile() && WORK_DIR.exists() && WORK_DIR.isDirectory()) {
          try {
            IOUtils.cp(keyFile, new File(Props.WORK_DIR.getPath() + File.separator + "key.tab"));
          } catch (IOException e) {
            warn(e);
          }
        }
      }
    } else {
      log(YELLOW, BOLD, "---------------------");
      log(YELLOW, BOLD, "   LIVE EVALUATION   ");
      log(YELLOW, BOLD, "---------------------");
      TEST_RESPONSES = Maybe.Nothing();
    }
    if (VALIDATION_QUERIES_BY_YEAR.containsKey(KBP_YEAR.name())) {
      VALIDATION_QUERIES = Maybe.Just(new File(VALIDATION_QUERIES_BY_YEAR.get(KBP_YEAR.name())));
    } else {
      if (Props.KBP_VALIDATE) { throw new IllegalArgumentException("Cannot run KBP in validate mode without validation queries for year " + KBP_YEAR); }
      VALIDATION_QUERIES = Maybe.Nothing();
    }
    // Initialize Features
    for (String feature : TRAIN_JOINTBAYES_YFEATURES_INTERNAL) {
      TRAIN_JOINTBAYES_YFEATURES.add(Y_FEATURE_CLASS.valueOf(feature.toUpperCase()));
    }
    if (TRAIN_JOINTBAYES_YFEATURES.isEmpty()) {
      warn("No training features specified!");
      //throw new IllegalStateException("No y features specified for JointBayes classifier");
    }
    if (TRAIN_FEATURES.length == 0) {
      warn("No training features specified!");
      //throw new IllegalStateException("No training features specified!");
    }
    // Initialize Thresholds
    for (Map.Entry<String, String> entry : TEST_THRESHOLD_JOINTBAYES_PERRELATION_IMPL.entrySet()) {
      TEST_THRESHOLD_JOINTBAYES_PERRELATION.put(RelationType.fromString(entry.getKey()).orCrash().canonicalName, Double.parseDouble(entry.getValue()));
    }
    // Check official index
    switch (Props.KBP_YEAR) {
      case KBP2009: assert Props.INDEX_OFFICIAL.getPath().contains("tac-kbp-2010-source-annotated"); break;
      case KBP2010: assert Props.INDEX_OFFICIAL.getPath().contains("tac-kbp-2010-source-annotated"); break;
      case KBP2011: assert Props.INDEX_OFFICIAL.getPath().contains("tac-kbp-2010-source-annotated"); break;
      case KBP2012: assert Props.INDEX_OFFICIAL.getPath().contains("tac-kbp-2010-source-annotated"); break;
      case KBP2013: assert Props.INDEX_OFFICIAL.getPath().contains("tac-kbp-2013-source-annotated"); break;
      case KBP2014: assert Props.INDEX_OFFICIAL.getPath().contains("tac-kbp-2013-source-annotated"); break;
    }
    try {
      Props.INDEX_OFFICIAL = Props.INDEX_OFFICIAL.getCanonicalFile();
    } catch (IOException e) {
      throw new IllegalArgumentException("Could not find index at: " + Props.INDEX_OFFICIAL);
    }
    // Check valid run mode
    if (Props.KBP_VALIDATE && (Props.KBP_EVALUATE || Props.KBP_TRAIN)) {
      throw new IllegalArgumentException("Cannot evaluate or train if kbp.validate is set");
    }

    // Create entity Linker
    Props.ENTITYLINKING_LINKER = new Lazy<EntityLinker>() {
      @Override
      protected EntityLinker compute() {
        if(KBP_ENTITYLINKER_CLASS.equals(TrainedEntityLinker.class)) {
          log("reading serialized entity linker from " + Props.ENTITYLINKING_PATHTOLINKER);
          return IOUtils.readObjectFromFileNoExceptions(ENTITYLINKING_PATHTOLINKER);
        }
        else {
          return MetaClass.create(Props.KBP_ENTITYLINKER_CLASS).createInstance();
        }
      }
    };
    Props.ENTITYLINKING_WIKIBACKOFF = new Lazy<EntityLinker>() {
      @Override
      protected EntityLinker compute() {
        if(ENTITYLINKING_WIKIBACKOFF_CLASS.equals(TrainedEntityLinker.class)) {
          log("reading serialized entity linker from " + Props.ENTITYLINKING_PATHTOLINKER);
          return IOUtils.readObjectFromFileNoExceptions(ENTITYLINKING_PATHTOLINKER);
        }
        else {
          return MetaClass.create(Props.ENTITYLINKING_WIKIBACKOFF_CLASS).createInstance();
        }
      }
    };

    // Create rules
    if(Props.TEST_GRAPH_INFERENCE_DO || Props.JUNIT) {
      Props.TEST_GRAPH_INFERENCE_RULES_FILES = CollectionUtils.map(
              Arrays.asList(Props.TEST_GRAPH_INFERENCE_RULES.split(",")),
          in -> {
            in = in.trim();
            assert IOUtils.existsInClasspathOrFileSystem(in);
            IOUtils.existsInClasspathOrFileSystem(in);
            return new File(in);
          }
      );
    }
  }

  /**
   * The default options declared in this class, before they have been overwritten by {@link edu.stanford.nlp.util.Execution}.
   */
  public static final Map<String, String> defaultOptions = Collections.unmodifiableMap(asMap());

  /**
   * Get all the options in this class as a map from the option name (as per the {@link Option} annotation),
   * and the {@link Object#toString()} method of the implementation of the object.
   */
  public static Map<String, String> asMap() {
    Map<String, String> map = new HashMap<>();
    Field[] fields = Props.class.getFields();
    for (Field field : fields) {
      Option o = field.getAnnotation(Option.class);
      if (o != null) {
        try {
          boolean savedPermission = field.isAccessible();
          if (!savedPermission) { field.setAccessible(true); }
          Object value = field.get(null);
          if (value != null) {
            if (value.getClass().isArray()) {
              map.put(o.name(), Arrays.toString( (Object[]) value ));
            } else {
              map.put(o.name(), value.toString());
            }
          } else {
            map.put(o.name(), null);
          }
          if (!savedPermission) { field.setAccessible(savedPermission); }
        } catch (IllegalAccessException e) {
          warn("could not read field: " + field);
        }
      }
    }
    return map;
  }



  //
  // CONSTANTS
  //
  public static final String ANNOTATORS = "tokenize, ssplit, pos, lemma, ner, regexner, parse, dcoref";
  /** when tuning slot thresholds fails (due to little/no data), we fall back to this threshold value */
  public static final double DEFAULT_SLOT_THRESHOLD = 0.50;
  /** Maximum distance between entity and slot candidate, in tokens */
  public static final int MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT = 1000;
  public static final String NER_BLANK_STRING = SeqClassifierFlags.DEFAULT_BACKGROUND_SYMBOL;
  /** Softmax parameter for relation classifier */
  public static final double SOFTMAX_GAMMA = 1.0;
  /** Extension for the serialized extractor models */
  public static final String SER_EXT = ".ser";

  /** The table name for the datum cache */
  public static final String DB_TABLE_DATUM_CACHE = "datum_cache";
  /** The table name for the sentence cache */
  public static final String DB_TABLE_SENTENCE_CACHE = "sentence_cache";
  /** The table name for the document cache */
  public static final String DB_TABLE_DOCUMENT_CACHE = "document_cache";
  /** The table name for the sentence gloss cache */
  public static final String DB_TABLE_SENTENCEGLOSS_CACHE = "sentencegloss_cache";
  /** The table name for the graph cache */
  public static final String DB_TABLE_GRAPH_CACHE = "graph_cache";
  /** The table name for the provenance cache */
  public static final String DB_TABLE_PROVENANCE_CACHE = "provenance_cache";
  /** The table name for the provenance cache */
  public static final String DB_TABLE_MINED_FORMULAS = "mined_formulas";
  /** The table name for entityContexts */
  public static final String DB_TABLE_ENTITY_CONTEXT = "entity_contexts";
  /** The table name with all entity linking data */
  public static final String DB_TABLE_ENTITYLINKING_DATA = "entitylinking_data";
  /** The table name with all wikidict positives */
  public static final String DB_TABLE_ENTITYLINKING_WIKI_POS = "entitylinking_wikidict_positive";
  /** The table name with all wikidict negatives */
  public static final String DB_TABLE_ENTITYLINKING_WIKI_NEG = "entitylinking_wikidict_negative";
  /** A holder for the main execution thread; or, at least the thread that instantiated Props (usually the main thread) */
  public static final Thread MAIN_THREAD = Thread.currentThread();

}
