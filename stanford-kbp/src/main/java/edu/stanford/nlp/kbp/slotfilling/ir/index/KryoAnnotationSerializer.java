package edu.stanford.nlp.kbp.slotfilling.ir.index;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;
import com.esotericsoftware.kryo.serializers.CompatibleFieldSerializer;
import com.esotericsoftware.kryo.serializers.FieldSerializer;
import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.*;
import edu.stanford.nlp.dcoref.Dictionaries;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.*;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.time.Timex;
import edu.stanford.nlp.trees.TreeCoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;
import org.objenesis.strategy.SerializingInstantiatorStrategy;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static edu.stanford.nlp.util.logging.Redwood.Util.err;

/**
 * A custom annotator, using the Kryo framework.
 * A fair bit of the code, particularly for serializing Semantic Graphs,
 * was taken from CustomAnnotationSerializer and adapted for the Kryo
 * framework.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("unchecked")
public class KryoAnnotationSerializer extends AnnotationSerializer {

  private final Kryo kryo;
  private final boolean compress;
  private final boolean includeDependencyRoots;

  private static final boolean DEFAULT_COMPRESS  = true;
  private static final boolean DEFAULT_ROBUST    = true;  // Was false for 2013 evaluation
  private static final boolean DEFAULT_SAVEROOTS = true;  // Was false for 2013 evaluation

  public String toString() {
    return "KryoAnnotationSerializer(" +
            "compress=" + compress +
            ", includeDependencyRoots=" + includeDependencyRoots + ")";
  }

  public KryoAnnotationSerializer(String name, Properties properties) {
    this(
      PropertiesUtils.getBool(properties, name + ".compress", DEFAULT_COMPRESS),
      PropertiesUtils.getBool(properties, name + ".robust", DEFAULT_ROBUST),
      PropertiesUtils.getBool(properties, name + ".includeDependencyRoots", DEFAULT_SAVEROOTS)
    );
  }

  public KryoAnnotationSerializer() {
    this(DEFAULT_COMPRESS, DEFAULT_ROBUST, DEFAULT_SAVEROOTS);
  }

  public KryoAnnotationSerializer(boolean compress, boolean robustBackwardsCompatibility) {
    this(compress, robustBackwardsCompatibility, DEFAULT_SAVEROOTS);
  }

  public KryoAnnotationSerializer(boolean compress, boolean robustBackwardsCompatibility, final boolean includeDependencyRoots) {
    this.kryo = new Kryo();
    this.compress = compress;
    this.includeDependencyRoots = includeDependencyRoots;

    // Trees are not really collections, and this causes Kryo to lose
    // its marbles: http://i.imgur.com/FSakhIy.gif.
    // Also, for some reason this has to come before the register() calls
    // below?
    kryo.addDefaultSerializer(Collection.class, new CollectionSerializer() {
      private final FieldSerializer<Tree> treeSerializer = new FieldSerializer<Tree>(kryo, Tree.class);

      @Override
      public void write(Kryo kryo, Output output, Collection collection) {
        assert collection != null;
        for (Object o : collection) { assert o != null; }
        if (collection instanceof LabeledScoredTreeNode) {
          Tree tree = (LabeledScoredTreeNode) collection;
          String treeString = tree.toString();
          assert treeString != null;
          byte[] bytes = treeString.getBytes();
          output.writeInt(bytes.length);
          output.write(bytes);
        } else if (collection instanceof Tree) {
          treeSerializer.write(kryo, output, (Tree) collection);
        } else if (unmodifiableListClass.isAssignableFrom(collection.getClass())) {
          ArrayList copy = new ArrayList();
          copy.addAll(collection);
          super.write(kryo, output, copy);
        } else {
          super.write(kryo, output, collection);
        }
      }

      @Override
      public Collection read(Kryo kryo, Input input, Class<Collection> collectionClass) {
        // NOTE: If you're reading a null tree, you're going to have a bad time.
        // The collectionClass will recognize that it should be a tree, but the writer had
        // no way of knowing that you were writing a tree, so it wrote the tree as a default
        // collection.
        // I blame Java for not having real type safety,
        // but if you find that impractical you can go ahead and blame me too.
        if (LabeledScoredTreeNode.class.isAssignableFrom(collectionClass)) {
          int length = input.readInt();
          String treeString = new String(input.readBytes(length));
          try {
            return new PennTreeReader(new StringReader(treeString), new LabeledScoredTreeFactory(CoreLabel.factory())).readTree();
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        } else if (Tree.class.isAssignableFrom(collectionClass)) {
          //noinspection unchecked
          return treeSerializer.read(kryo, input, (Class) collectionClass);
        } else if (unmodifiableListClass.isAssignableFrom(collectionClass)) {
          return Collections.unmodifiableList((List) super.read(kryo, input, (Class) ArrayList.class));
        } else {
          return super.read(kryo, input, collectionClass);
        }
      }
    });

    // GrammaticalRelation is a funky class to serialize (multithreaded at least)
    kryo.addDefaultSerializer(SemanticGraph.class, new Serializer<SemanticGraph>(){
      @Override
      public void write(Kryo kryo, Output output, SemanticGraph graph) {
        // Is Null
        if (graph == null) {
          output.writeBoolean(true);
          return;
        } else {
          output.writeBoolean(false);
        }

        // Meta-info
        Set<IndexedWord> nodes = graph.vertexSet();
        output.writeInt(nodes.size());
        if (!nodes.isEmpty()) {
          IndexedWord node = nodes.iterator().next();
          assert node.containsKey(DocIDAnnotation.class);
          output.writeString(node.get(DocIDAnnotation.class));
          assert node.containsKey(SentenceIndexAnnotation.class);
          output.writeInt(node.get(SentenceIndexAnnotation.class));
        }

        // Token Info
        for (IndexedWord node : graph.vertexSet()) {
          output.writeInt(node.index());
          if (false) {  // was: copy annotation
            output.writeBoolean(true);
//            output.writeInt(node.get(CopyAnnotation.class));
          } else {
            output.writeBoolean(false);
          }
        }

        // Edges
        for (SemanticGraphEdge edge : graph.edgeIterable()) {
          assert edge != null && edge.getRelation() != null && edge.getRelation().toString() != null;
          String rel = edge.getRelation().toString();
          int source = edge.getSource().index();
          int target = edge.getTarget().index();
          boolean extra = edge.isExtra();
          output.writeBoolean(true);
          output.writeString(rel);
          output.writeInt(source);
          output.writeInt(target);
          output.writeBoolean(extra);
        }
        output.writeBoolean(false);

        if (includeDependencyRoots) {
          // Roots (not necessary part of the vertices)
          Collection<IndexedWord> roots = graph.getRoots();
          assert roots != null;
          output.writeInt(roots.size());
          if (nodes.isEmpty()) {
            // there was no DocID or sentenceIndex information!!!
            // Have some here
            if (!roots.isEmpty()) {
              IndexedWord node = roots.iterator().next();
              assert node.containsKey(DocIDAnnotation.class);
              output.writeString(node.get(DocIDAnnotation.class));
              assert node.containsKey(SentenceIndexAnnotation.class);
              output.writeInt(node.get(SentenceIndexAnnotation.class));
            }
          }
          for (IndexedWord node : roots) {
            output.writeInt(node.index());
            if (false) {  // was: copy annotation
              output.writeBoolean(true);
//              output.writeInt(node.get(CopyAnnotation.class));
            } else {
              output.writeBoolean(false);
            }
          }
        }
      }

      @Override
      public SemanticGraph read(Kryo kryo, Input input, Class<SemanticGraph> grammaticalRelationClass) {
        // Is null
        boolean isNull = input.readBoolean();
        if (isNull) return null;

        // Meta-info
        int size = input.readInt();
        String docid = size > 0 ? input.readString() : "";
        int sentenceIndex = size > 0 ? input.readInt() : -1;

        // Token Info
        List<IntermediateNode> nodes = new ArrayList<IntermediateNode>(size);
        for ( int i = 0; i < size; ++i) {
          int index = input.readInt();
          Integer copy = null;
          if (input.readBoolean()) { copy = input.readInt(); }
          nodes.add(new IntermediateNode(docid, sentenceIndex, index, copy == null ? -1 : copy));
        }

        // Edges
        List<IntermediateEdge> edges = new ArrayList<IntermediateEdge>(size + 1);
        while (input.readBoolean()) {
          String rel = input.readString();
          int source = input.readInt();
          int target = input.readInt();
          boolean isExtra = input.readBoolean();
          edges.add(new IntermediateEdge(rel, source, target, isExtra));
        }

        List<IntermediateNode> roots = null;
        if (includeDependencyRoots) {
          // Crazy dependency roots - why aren't they part of the vertices
          int nRoots = input.readInt();
          roots = new ArrayList<IntermediateNode>(nRoots);
          if (nodes.isEmpty()) {
            // there was no DocID or sentenceIndex information!!!
            // Have some here
            if (nRoots > 0) {
              docid = input.readString();
              sentenceIndex = input.readInt();
            }
          }
          for ( int i = 0; i < nRoots; ++i) {
            int index = input.readInt();
            Integer copy = null;
            if (input.readBoolean()) { copy = input.readInt(); }
            roots.add(new IntermediateNode(docid, sentenceIndex, index, copy == null ? -1 : copy));
          }
        }
        return new IntermediateSemanticGraph(nodes, edges, roots);
      }
    });

    // IMPORTANT NOTE: Add new classes to the *END* of this list,
    // and don't change these numbers.
    // otherwise, de-serializing existing serializations won't work.
    // Note that registering classes here is an efficiency tweak, and not
    // strictly necessary
    kryo.register(String.class, 0);
    kryo.register(Short.class, 1);
    kryo.register(Integer.class, 2);
    kryo.register(Long.class, 3);
    kryo.register(Float.class, 4);
    kryo.register(Double.class, 5);
    kryo.register(Boolean.class, 6);
    kryo.register(List.class, 17);
    kryo.register(ArrayList.class, 18);
    kryo.register(LinkedList.class, 19);
    kryo.register(ArrayCoreMap.class, 20);
    kryo.register(CoreMap.class, 21);
    kryo.register(CoreLabel.class, 22);
    kryo.register(Calendar.class, 23);
    kryo.register(Map.class, 24);
    kryo.register(HashMap.class, 25);
    kryo.register(Pair.class, 26);
    kryo.register(Tree.class, 27);
    kryo.register(LabeledScoredTreeNode.class, 28);
    kryo.register(TreeGraphNode.class, 29);
    kryo.register(CorefChain.class, 30);
    kryo.register(CorefChain.CorefMention.class, 32);
    kryo.register(Dictionaries.MentionType.class, 33);
    kryo.register(Dictionaries.Animacy.class, 34);
    kryo.register(Dictionaries.Gender.class, 35);
    kryo.register(Dictionaries.Number.class, 36);
    kryo.register(Dictionaries.Person.class, 37);
    kryo.register(Set.class, 38);
    kryo.register(HashSet.class, 39);
    kryo.register(Label.class, 40);
    kryo.register(SemanticGraph.class, 41);
    kryo.register(SemanticGraphEdge.class, 42);
    kryo.register(IndexedWord.class, 43);
    kryo.register(Timex.class, 44);
    kryo.register(subListClass, 45);
    // The keys (hey, why not)
    kryo.register(LabelWeightAnnotation.class, 101);
    kryo.register(AntecedentAnnotation.class, 102);
    kryo.register(LeftChildrenNodeAnnotation.class, 103);
    kryo.register(MentionTokenAnnotation.class, 104);
    kryo.register(ParagraphAnnotation.class, 105);
    kryo.register(SpeakerAnnotation.class, 106);
    kryo.register(UtteranceAnnotation.class, 107);
    kryo.register(UseMarkedDiscourseAnnotation.class, 108);
    kryo.register(NumerizedTokensAnnotation.class, 109);
    kryo.register(NumericCompositeObjectAnnotation.class, 110);
    kryo.register(NumericCompositeTypeAnnotation.class, 111);
    kryo.register(NumericCompositeValueAnnotation.class, 112);
    kryo.register(NumericObjectAnnotation.class, 113);
    kryo.register(NumericValueAnnotation.class, 114);
    kryo.register(NumericTypeAnnotation.class, 115);
    kryo.register(DocDateAnnotation.class, 116);
    kryo.register(CommonWordsAnnotation.class, 117);
    kryo.register(ProtoAnnotation.class, 118);
    kryo.register(PhraseWordsAnnotation.class, 119);
    kryo.register(PhraseWordsTagAnnotation.class, 120);
    kryo.register(WordnetSynAnnotation.class, 121);
    kryo.register(TopicAnnotation.class, 122);
    kryo.register(XmlContextAnnotation.class, 123);
    kryo.register(XmlElementAnnotation.class, 124);
//    kryo.register(CopyAnnotation.class, 125);
    kryo.register(ArgDescendentAnnotation.class, 126);
    kryo.register(CovertIDAnnotation.class, 127);
    kryo.register(SemanticTagAnnotation.class, 128);
    kryo.register(SemanticWordAnnotation.class, 129);
    kryo.register(PriorAnnotation.class, 130);
    kryo.register(YearAnnotation.class, 131);
    kryo.register(DayAnnotation.class, 132);
    kryo.register(MonthAnnotation.class, 133);
    kryo.register(HeadWordStringAnnotation.class, 134);
    kryo.register(GrandparentAnnotation.class, 135);
    kryo.register(PercentAnnotation.class, 136);
    kryo.register(NotAnnotation.class, 137);
    kryo.register(BeAnnotation.class, 138);
    kryo.register(HaveAnnotation.class, 139);
    kryo.register(DoAnnotation.class, 140);
    kryo.register(UnaryAnnotation.class, 141);
    kryo.register(FirstChildAnnotation.class, 142);
    kryo.register(PrevChildAnnotation.class, 143);
    kryo.register(StateAnnotation.class, 144);
    kryo.register(SpaceBeforeAnnotation.class, 145);
    kryo.register(UBlockAnnotation.class, 146);
    kryo.register(D2_LEndAnnotation.class, 147);
    kryo.register(D2_LMiddleAnnotation.class, 148);
    kryo.register(D2_LBeginAnnotation.class, 149);
    kryo.register(LEndAnnotation.class, 150);
    kryo.register(LMiddleAnnotation.class, 151);
    kryo.register(LBeginAnnotation.class, 152);
    kryo.register(LengthAnnotation.class, 153);
    kryo.register(HeightAnnotation.class, 154);
    kryo.register(BagOfWordsAnnotation.class, 155);
    kryo.register(SubcategorizationAnnotation.class, 156);
    kryo.register(TrueTagAnnotation.class, 157);
    kryo.register(WordFormAnnotation.class, 158);
    kryo.register(DependentsAnnotation.class, 159);
    kryo.register(ContextsAnnotation.class, 160);
    kryo.register(NeighborsAnnotation.class, 161);
    kryo.register(LabelAnnotation.class, 162);
    kryo.register(LastTaggedAnnotation.class, 163);
    kryo.register(BestFullAnnotation.class, 164);
    kryo.register(BestCliquesAnnotation.class, 165);
    kryo.register(AnswerObjectAnnotation.class, 166);
    kryo.register(EntityClassAnnotation.class, 167);
    kryo.register(SentenceIDAnnotation.class, 168);
    kryo.register(SentencePositionAnnotation.class, 169);
    kryo.register(ParaPositionAnnotation.class, 170);
    kryo.register(WordPositionAnnotation.class, 171);
    kryo.register(SectionAnnotation.class, 172);
    kryo.register(EntityRuleAnnotation.class, 173);
    kryo.register(UTypeAnnotation.class, 174);
    kryo.register(OriginalCharAnnotation.class, 175);
    kryo.register(OriginalAnswerAnnotation.class, 176);
    kryo.register(PredictedAnswerAnnotation.class, 177);
    kryo.register(IsDateRangeAnnotation.class, 178);
    kryo.register(EntityTypeAnnotation.class, 179);
    kryo.register(IsURLAnnotation.class, 180);
    kryo.register(LastGazAnnotation.class, 181);
    kryo.register(MaleGazAnnotation.class, 182);
    kryo.register(FemaleGazAnnotation.class, 183);
    kryo.register(WebAnnotation.class, 184);
    kryo.register(DictAnnotation.class, 185);
    kryo.register(FreqAnnotation.class, 186);
    kryo.register(AbstrAnnotation.class, 187);
    kryo.register(GeniaAnnotation.class, 188);
    kryo.register(AbgeneAnnotation.class, 189);
    kryo.register(GovernorAnnotation.class, 190);
    kryo.register(ChunkAnnotation.class, 191);
    kryo.register(AbbrAnnotation.class, 192);
    kryo.register(DistSimAnnotation.class, 193);
    kryo.register(PossibleAnswersAnnotation.class, 194);
    kryo.register(GazAnnotation.class, 195);
    kryo.register(IDAnnotation.class, 196);
    kryo.register(UnknownAnnotation.class, 197);
    kryo.register(CharAnnotation.class, 198);
    kryo.register(PositionAnnotation.class, 199);
    kryo.register(DomainAnnotation.class, 200);
    kryo.register(TagLabelAnnotation.class, 201);
    kryo.register(NumTxtSentencesAnnotation.class, 202);
    kryo.register(SRLInstancesAnnotation.class, 203);
    kryo.register(WordSenseAnnotation.class, 204);
    kryo.register(CostMagnificationAnnotation.class, 205);
    kryo.register(CharacterOffsetEndAnnotation.class, 206);
    kryo.register(CharacterOffsetBeginAnnotation.class, 207);
    kryo.register(ChineseIsSegmentedAnnotation.class, 208);
    kryo.register(ChineseSegAnnotation.class, 209);
    kryo.register(ChineseOrigSegAnnotation.class, 210);
    kryo.register(ChineseCharAnnotation.class, 211);
    kryo.register(MorphoCaseAnnotation.class, 212);
    kryo.register(MorphoGenAnnotation.class, 213);
    kryo.register(MorphoPersAnnotation.class, 214);
    kryo.register(MorphoNumAnnotation.class, 215);
    kryo.register(PolarityAnnotation.class, 216);
    kryo.register(StemAnnotation.class, 217);
    kryo.register(GazetteerAnnotation.class, 218);
    kryo.register(RoleAnnotation.class, 219);
    kryo.register(InterpretationAnnotation.class, 220);
    kryo.register(FeaturesAnnotation.class, 221);
    kryo.register(GoldAnswerAnnotation.class, 222);
    kryo.register(AnswerAnnotation.class, 223);
    kryo.register(SpanAnnotation.class, 224);
    kryo.register(INAnnotation.class, 225);
    kryo.register(ParentAnnotation.class, 226);
    kryo.register(LeftTermAnnotation.class, 227);
    kryo.register(ShapeAnnotation.class, 228);
    kryo.register(SRLIDAnnotation.class, 229);
    kryo.register(SRL_ID.class, 230);
    kryo.register(NormalizedNamedEntityTagAnnotation.class, 231);
    kryo.register(NERIDAnnotation.class, 232);
    kryo.register(CategoryFunctionalTagAnnotation.class, 233);
    kryo.register(VerbSenseAnnotation.class, 234);
    kryo.register(SemanticHeadTagAnnotation.class, 235);
    kryo.register(SemanticHeadWordAnnotation.class, 236);
    kryo.register(MarkingAnnotation.class, 237);
    kryo.register(ArgumentAnnotation.class, 238);
    kryo.register(ProjectedCategoryAnnotation.class, 239);
    kryo.register(IDFAnnotation.class, 240);
    kryo.register(CoNLLDepParentIndexAnnotation.class, 241);
    kryo.register(CoNLLDepTypeAnnotation.class, 242);
    kryo.register(CoNLLSRLAnnotation.class, 243);
    kryo.register(CoNLLPredicateAnnotation.class, 244);
    kryo.register(CoNLLDepAnnotation.class, 245);
    kryo.register(CoarseTagAnnotation.class, 246);
    kryo.register(AfterAnnotation.class, 247);
    kryo.register(BeforeAnnotation.class, 248);
    kryo.register(OriginalTextAnnotation.class, 249);
    kryo.register(CategoryAnnotation.class, 250);
    kryo.register(ValueAnnotation.class, 251);
    kryo.register(LineNumberAnnotation.class, 252);
    kryo.register(SentenceIndexAnnotation.class, 253);
    kryo.register(ForcedSentenceEndAnnotation.class, 254);
    kryo.register(EndIndexAnnotation.class, 255);
    kryo.register(BeginIndexAnnotation.class, 256);
    kryo.register(IndexAnnotation.class, 257);
    kryo.register(DocIDAnnotation.class, 258);
    kryo.register(CalendarAnnotation.class, 259);
    kryo.register(TokenEndAnnotation.class, 260);
    kryo.register(TokenBeginAnnotation.class, 261);
    kryo.register(ParagraphsAnnotation.class, 262);
    kryo.register(SentencesAnnotation.class, 263);
    kryo.register(GenericTokensAnnotation.class, 264);
    kryo.register(TokensAnnotation.class, 265);
    kryo.register(TrueCaseTextAnnotation.class, 266);
    kryo.register(TrueCaseAnnotation.class, 267);
    kryo.register(StackedNamedEntityTagAnnotation.class, 268);
    kryo.register(NamedEntityTagAnnotation.class, 269);
    kryo.register(PartOfSpeechAnnotation.class, 270);
    kryo.register(LemmaAnnotation.class, 271);
    kryo.register(TextAnnotation.class, 272);
    kryo.register(CorefChainAnnotation.class, 273);
    kryo.register(CorefClusterAnnotation.class, 274);
    kryo.register(CorefClusterIdAnnotation.class, 275);
    //noinspection deprecation
    kryo.register(CorefGraphAnnotation.class, 276);
    kryo.register(CorefDestAnnotation.class, 277);
    kryo.register(CorefAnnotation.class, 278);
    kryo.register(HeadTagAnnotation.class, 279);
    kryo.register(HeadWordAnnotation.class, 280);
    kryo.register(TreeAnnotation.class, 281);
    kryo.register(CollapsedDependenciesAnnotation.class, 282);
    kryo.register(CollapsedCCProcessedDependenciesAnnotation.class, 283);
    kryo.register(TimeAnnotations.TimexAnnotation.class, 284);
    kryo.register(TimeAnnotations.TimexAnnotations.class, 285);

    // Handle zero argument constructors gracefully
    kryo.setInstantiatorStrategy(new SerializingInstantiatorStrategy());

    // Robust backwards compatibility.
    // Fields can be added or removed, but their signature
    // can't be changed (this seems fair).
    if (robustBackwardsCompatibility) {
      kryo.setDefaultSerializer(CompatibleFieldSerializer.class);
    }
  }

  @Override
  public synchronized OutputStream write(Annotation corpus, OutputStream os) throws IOException {
    if (os instanceof Output) {
      this.kryo.writeObject((Output) os, corpus);
      return os;
    } else {
      OutputStream actualOutputStream = (compress && !(os instanceof GZIPOutputStream)) ? new GZIPOutputStream(os) : os;
      Output output = new Output(actualOutputStream);
      this.kryo.writeObject(output, corpus);
      return output;
    }
  }

  @Override
  public synchronized Pair<Annotation, InputStream> read(InputStream is) throws IOException, ClassNotFoundException, ClassCastException {
    // Read Annotation
    Input input;
    if (is instanceof Input) {
      input = (Input) is;
    } else {
      input = new Input((compress && !(is instanceof GZIPInputStream)) ? new GZIPInputStream(is) : is);
    }
    Annotation someObject = kryo.readObject(input, Annotation.class);
    // Fix Annotation
    fixAnnotation(someObject);
    // Return
    return Pair.makePair(someObject, (InputStream) input);
  }

  /**
   * Create a new FileBackedCache over Annotations, using the Kryo serialization framework
   * as a backend.
   * @param directory The directory to use for the cache.
   * @param numFiles The maximum number of files to have in the cache
   * @param <KEY> The type of the key we are caching on
   * @return A new FileBackedCache, but with Kryo plugged into the backend
   */
  public <KEY extends Serializable> FileBackedCache<KEY, Annotation> createCache(File directory, int numFiles, final boolean doFileLock) {
    return new FileBackedCache<KEY, Annotation>(directory, numFiles) {
      @Override
      protected Pair<? extends InputStream, CloseAction> newInputStream(File f) throws IOException {
        synchronized (kryo) {
          final FileSemaphore lock = doFileLock ? acquireFileLock(f) : null;
          final InputStream rtn = new Input(compress ? new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))) : new BufferedInputStream(new FileInputStream(f)));
          return new Pair<InputStream, CloseAction>(rtn,
              () -> { synchronized (kryo) { if (lock != null) { lock.release(); } rtn.close(); } });
        }
      }

      @Override
      protected Pair<? extends OutputStream, CloseAction> newOutputStream(File f, boolean isAppend) throws IOException {
        synchronized (kryo) {
          final FileSemaphore lock = doFileLock ? acquireFileLock(f) : null;
          final FileOutputStream stream = new FileOutputStream(f, isAppend);
          final OutputStream rtn = new Output(compress ? new GZIPOutputStream(new BufferedOutputStream(stream)) : new BufferedOutputStream(stream));
          return new Pair<OutputStream, CloseAction>(rtn,
              () -> { synchronized (kryo) { rtn.flush(); if (lock != null) { lock.release(); } rtn.close(); } });
        }
      }

      @Override
      protected Pair<KEY, Annotation> readNextObjectOrNull(InputStream input) throws IOException, ClassNotFoundException {
        synchronized (kryo) {
          if ( ((Input) input).canReadInt() ) {
            if ( ((Input) input).readInt() != 42 ) {
              throw new IllegalStateException("Kryo cache doesn't have header for object");
            }
            try {
              Pair<KEY, Annotation> pair = kryo.readObject((Input) input, Pair.class);
              fixAnnotation(pair.second);
              return pair;
            } catch (KryoException e) {
              err("caught exception; printing and returning null...");
              err(e);
              return null;
            }
          } else {
            return null;
          }
        }
      }

      @Override
      protected synchronized void writeNextObject(OutputStream output, Pair<KEY, Annotation> value) throws IOException {
        synchronized (kryo) {
          ((Output) output).writeInt(42);
          kryo.writeObject((Output) output, value);
        }
      }
    };
  }

  /**
   * Create a new FileBackedCache over Annotations, using the Kryo serialization framework
   * as a backend.
   * @param directory The directory to use for the cache.
   * @param numFiles The maximum number of files to have in the cache
   * @param <KEY> The type of the key we are caching on
   * @return A new FileBackedCache, but with Kryo plugged into the backend
   */
  public <KEY extends Serializable> FileBackedCache<KEY, ArrayList<Annotation>> createDocumentCache(File directory, int numFiles, final boolean doFileLock) {
    return new FileBackedCache<KEY, ArrayList<Annotation>>(directory, numFiles) {
      @Override
      protected Pair<? extends InputStream, CloseAction> newInputStream(File f) throws IOException {
        synchronized (kryo) {
          final FileSemaphore lock = doFileLock ? acquireFileLock(f) : null;
          final InputStream rtn = new Input(compress ? new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))) : new BufferedInputStream(new FileInputStream(f)));
          return new Pair<InputStream, CloseAction>(rtn,
              () -> { synchronized (kryo) { if (lock != null) { lock.release(); } rtn.close(); } });
        }
      }

      @Override
      protected Pair<? extends OutputStream, CloseAction> newOutputStream(File f, boolean isAppend) throws IOException {
        synchronized (kryo) {
          final FileSemaphore lock = doFileLock ? acquireFileLock(f) : null;
          final FileOutputStream stream = new FileOutputStream(f, isAppend);
          final OutputStream rtn = new Output(compress ? new GZIPOutputStream(new BufferedOutputStream(stream)) : new BufferedOutputStream(stream));
          return new Pair<OutputStream, CloseAction>(rtn,
              () -> { synchronized (kryo) { rtn.flush(); if (lock != null) { lock.release(); } rtn.close(); } });
        }
      }

      @Override
      protected Pair<KEY, ArrayList<Annotation>> readNextObjectOrNull(InputStream input) throws IOException, ClassNotFoundException {
        synchronized (kryo) {
          if ( ((Input) input).canReadInt() ) {
            if ( ((Input) input).readInt() != 42 ) {
              throw new IllegalStateException("Kryo cache doesn't have header for object");
            }
            Pair<KEY, ArrayList<Annotation>> pair = kryo.readObject((Input) input, Pair.class);
            for( Annotation ann : pair.second ) fixAnnotation(ann);
            return pair;
          } else {
            return null;
          }
        }
      }

      @Override
      protected synchronized void writeNextObject(OutputStream output, Pair<KEY, ArrayList<Annotation>> value) throws IOException {
        synchronized (kryo) {
          ((Output) output).writeInt(42);
          kryo.writeObject((Output) output, value);
        }
      }
    };
  }



  private static final Class unmodifiableListClass;
  private static final Class subListClass;

  static {
    ArrayList<Integer> dummy = new ArrayList<Integer>();
    dummy.add(1);
    dummy.add(2);
    subListClass = dummy.subList(0, 1).getClass();
    unmodifiableListClass = Collections.unmodifiableList(dummy).getClass();
  }

  private static void fixAnnotation(Annotation someObject) {
    // Fix up Annotation (e.g., dependency graph)
    if (someObject.get(SentencesAnnotation.class) != null) {
      for (CoreMap sentence : someObject.get(SentencesAnnotation.class)) {
        List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
        if (sentence.containsKey(CollapsedDependenciesAnnotation.class)) {
          sentence.set(CollapsedDependenciesAnnotation.class,
              convertIntermediateGraph((IntermediateSemanticGraph) sentence.get(CollapsedDependenciesAnnotation.class), tokens));
        }
        if (sentence.containsKey(BasicDependenciesAnnotation.class)) {
          sentence.set(BasicDependenciesAnnotation.class,
              convertIntermediateGraph((IntermediateSemanticGraph) sentence.get(BasicDependenciesAnnotation.class), tokens));
        }
        if (sentence.containsKey(CollapsedCCProcessedDependenciesAnnotation.class)) {
          sentence.set(CollapsedCCProcessedDependenciesAnnotation.class,
              convertIntermediateGraph((IntermediateSemanticGraph) sentence.get(CollapsedCCProcessedDependenciesAnnotation.class), tokens));
        }
      }
    }
  }

  // TODO(gabor) this was factored out into AnnotationSerializer by John, but that broke KBP. ProtobufAnnotationSerializer is better moving forward, so this class is perhaps deprecated  anyways.
  private static SemanticGraph convertIntermediateGraph(IntermediateSemanticGraph ig, List<CoreLabel> sentence) {
    SemanticGraph graph = new SemanticGraph();

    // first construct the actual nodes; keep them indexed by their index
    // This block is optimized as one of the places which take noticeable time
    // in datum caching
    int min = Integer.MAX_VALUE;
    int max = Integer.MIN_VALUE;
    for(IntermediateNode in: ig.nodes){
      min = in.index < min ? in.index : min;
      max = in.index > max ? in.index : max;
    }
    IndexedWord[] nodes = new IndexedWord[max - min >= 0 ? max - min + 1 : 0];
    for(IntermediateNode in: ig.nodes){
      CoreLabel token = sentence.get(in.index - 1); // index starts at 1!
      token.set(DocIDAnnotation.class, in.docId);
      token.set(SentenceIndexAnnotation.class, in.sentIndex);
      token.set(IndexAnnotation.class, in.index);
      IndexedWord word = new IndexedWord(token);
      word.set(ValueAnnotation.class, word.get(TextAnnotation.class));
      if(in.copyAnnotation >= 0){
//        word.set(CopyAnnotation.class, in.copyAnnotation);
      }
      assert in.index == word.index();
      nodes[in.index - min] = word;
    }
    for (IndexedWord node : nodes) {
      if (node != null) { graph.addVertex(node); }
    }

    // add all edges to the actual graph
    synchronized (GrammaticalRelation.class) {
      for(IntermediateEdge ie: ig.edges){
        IndexedWord source = nodes[ie.source - min];
        assert(source != null);
        IndexedWord target = nodes[ie.target - min];
        assert(target != null);
        // this is not thread-safe: there are static fields in GrammaticalRelation
        GrammaticalRelation rel = GrammaticalRelation.valueOf(ie.dep);
        graph.addEdge(source, target, rel, 1.0, ie.isExtra);
        // end not threadsafe part
      }
    }

    if (ig.roots != null) {
      Collection<IndexedWord> roots = new ArrayList<IndexedWord>();
      for(IntermediateNode in: ig.roots){
        CoreLabel token = sentence.get(in.index - 1); // index starts at 1!
        token.set(DocIDAnnotation.class, in.docId);
        token.set(SentenceIndexAnnotation.class, in.sentIndex);
        token.set(IndexAnnotation.class, in.index);
        IndexedWord word = new IndexedWord(token);
        word.set(ValueAnnotation.class, word.get(TextAnnotation.class));
        if(in.copyAnnotation >= 0){
//          word.set(CopyAnnotation.class, in.copyAnnotation);
        }
        assert in.index == word.index();
        roots.add(word);
      }
      graph.setRoots(roots);
    } else {
      // Roots were not saved away
      // compute root nodes if non-empty
      if(!graph.isEmpty()){
        graph.resetRoots();
      }
    }

    return graph;
  }

  public static class IntermediateSemanticGraph extends SemanticGraph {
    final List<IntermediateNode> nodes;
    final List<IntermediateEdge> edges;
    final List<IntermediateNode> roots;
    IntermediateSemanticGraph(List<IntermediateNode> nodes, List<IntermediateEdge> edges, List<IntermediateNode> roots) {
      this.nodes = nodes;
      this.edges = edges;
      this.roots = roots;
    }
  }

  public static class IntermediateNode {
    String docId;
    int sentIndex;
    int index;
    int copyAnnotation;
    IntermediateNode(String docId, int sentIndex, int index, int copy) {
      this.docId = docId;
      this.sentIndex = sentIndex;
      this.index = index;
      this.copyAnnotation = copy;
    }
  }

  public static class IntermediateEdge {
    int source;
    int target;
    String dep;
    boolean isExtra;
    IntermediateEdge(String dep, int source, int target, boolean isExtra) {
      this.dep = dep;
      this.source = source;
      this.target = target;
      this.isExtra = isExtra;
    }
  }
}
