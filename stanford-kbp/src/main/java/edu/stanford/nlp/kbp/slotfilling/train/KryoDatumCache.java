package edu.stanford.nlp.kbp.slotfilling.train;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.CompatibleFieldSerializer;
import com.esotericsoftware.kryo.serializers.FieldSerializer;
import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.Dictionaries;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.time.Timex;
import edu.stanford.nlp.trees.LabeledScoredTreeNode;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeGraphNode;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.FileBackedCache;
import edu.stanford.nlp.util.Pair;
import org.objenesis.strategy.SerializingInstantiatorStrategy;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static edu.stanford.nlp.util.logging.Redwood.Util.err;

/**
 * A FileBackedCache, but with serialization backed by the Kryo serializer.
 *
 * @author Gabor Angeli
 */
public class KryoDatumCache extends FileBackedCache<KBTriple, Map<KBPair, SentenceGroup>> {
  private static final Kryo kryo = new Kryo();

  static {
    kryo.addDefaultSerializer(SentenceGroup.class, new FieldSerializer<SentenceGroup>(kryo, SentenceGroup.class));
    kryo.addDefaultSerializer(Maybe.Just("hi").getClass(), new FieldSerializer(kryo, Maybe.Just("hi").getClass()));
    kryo.addDefaultSerializer(Maybe.Nothing().getClass(), new FieldSerializer(kryo, Maybe.Nothing().getClass()));
    kryo.addDefaultSerializer(NERTag.class, new Serializer<NERTag>() {
      @Override
      public void write(Kryo kryo, Output output, NERTag entityType) {
        output.writeString(entityType.name);
      }
      @Override
      public NERTag read(Kryo kryo, Input input, Class<NERTag> entityTypeClass) {
        String repr = input.readString();
        return NERTag.fromString(repr).orCrash();
      }
    });
    kryo.addDefaultSerializer(NERTag.class, new Serializer<NERTag>() {
      @Override
      public void write(Kryo kryo, Output output, NERTag fill) {
        output.writeString(fill.name);
      }
      @Override
      public NERTag read(Kryo kryo, Input input, Class<NERTag> entityTypeClass) {
        return NERTag.fromString(input.readString()).orCrash();
      }
    });
    kryo.addDefaultSerializer(RelationType.class, new Serializer<RelationType>() {
      @Override
      public void write(Kryo kryo, Output output, RelationType rel) {
        output.writeString(rel.canonicalName);
      }
      @Override
      public RelationType read(Kryo kryo, Input input, Class<RelationType> clazz) {
        return RelationType.fromString(input.readString()).get();
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
    kryo.register(NERTag.class, 45);
    kryo.register(KBPSlotFill.class, 46);
    kryo.register(NERTag.class, 48);
    kryo.register(RelationType.Cardinality.class, 49);
    kryo.register(Maybe.class, 50);
    kryo.register(SentenceGroup.class, 51);

    // Handle zero argument constructors gracefully
    kryo.setInstantiatorStrategy(new SerializingInstantiatorStrategy());
    kryo.setDefaultSerializer(CompatibleFieldSerializer.class);
  }

  public KryoDatumCache(File directoryToCacheIn, int maxFiles) {
    super(directoryToCacheIn, maxFiles);
  }

  @Override
  protected Pair<? extends InputStream, CloseAction> newInputStream(File f) throws IOException {
    final FileSemaphore lock = Props.CACHE_LOCK ? acquireFileLock(f) : null;
    final InputStream rtn = new Input(new GZIPInputStream(new BufferedInputStream(new FileInputStream(f))));
    return new Pair<InputStream, CloseAction>(rtn,
        () -> { if (lock != null) { lock.release(); } rtn.close(); });
  }

  @Override
  protected Pair<? extends OutputStream, CloseAction> newOutputStream(File f, boolean isAppend) throws IOException {
    final FileOutputStream stream = new FileOutputStream(f, isAppend);
    final FileSemaphore lock = Props.CACHE_LOCK ? acquireFileLock(f) : null;
    final OutputStream rtn = new Output(new GZIPOutputStream(new BufferedOutputStream(stream)));
    return new Pair<OutputStream, CloseAction>(rtn,
        () -> { if (lock != null) { lock.release(); } rtn.close(); });
  }

  @SuppressWarnings("unchecked")
  @Override
  protected Pair<KBTriple, Map<KBPair, SentenceGroup>> readNextObjectOrNull(InputStream input) throws IOException, ClassNotFoundException {
    try {
      Pair<KBTriple, Map<KBPair, SentenceGroup>> rtn;
      synchronized (globalLock) {
        rtn =  kryo.readObject((Input) input, Pair.class);
      }
      if (Utils.assertionsEnabled()) {
        Iterator<Entry<KBPair, SentenceGroup>> iter = rtn.second.entrySet().iterator();
        while (iter.hasNext()) {
          Entry<KBPair, SentenceGroup> entry = iter.next();
          if (entry.getValue().sentenceGlossKeys.isDefined()) {
            if (entry.getValue().sentenceGlossKeys.get().size() != entry.getValue().size()) {
              err("Sentence gloss key size doesn't match datum size (sentencegloss=" + entry.getValue().sentenceGlossKeys.get().size() + ", datum=" + entry.getValue().size() + ")");
              iter.remove();
            }
          }
        }
      }
      return rtn;
    } catch (KryoException e) {
      if (e.getCause() != null && e.getCause() instanceof EOFException) {
        return null;
      } else if (e.getMessage().equals("Buffer underflow.")) {
        return null;
      } else {
        throw e;
      }
    }
  }

  @Override
  protected void writeNextObject(OutputStream output, Pair<KBTriple, Map<KBPair, SentenceGroup>> value) throws IOException {
    assert value.first != null;
    assert value.second != null;
    if (Utils.assertionsEnabled()) {
      for (Entry<KBPair, SentenceGroup> entry : value.second.entrySet()) {
        if (entry.getValue().sentenceGlossKeys.isDefined()) {
          assert entry.getValue().sentenceGlossKeys.get().size() == entry.getValue().size();
        }
      }
    }
    synchronized (globalLock) {
      kryo.writeObject((Output) output, value);
    }
  }

  public static void save(Map<KBPair, SentenceGroup> datums, OutputStream os) throws IOException {
    Output output = new Output(new GZIPOutputStream(os));
    synchronized (globalLock) {
      kryo.writeObject(output, datums);
    }
    output.close();
  }

  public static void save(SentenceGroup datums, OutputStream os) throws IOException {
    Output output = new Output(new GZIPOutputStream(os));
    synchronized (globalLock) {
      kryo.writeObject(output, datums);
    }
    output.close();
  }

  public static SentenceGroup load(InputStream is) throws IOException, ClassNotFoundException, ClassCastException {
    // Read Annotation
    Input input = new Input(new GZIPInputStream(is));
    SentenceGroup someObject;
    synchronized (globalLock) {
      //noinspection unchecked
      someObject = kryo.readObject(input, SentenceGroup.class);
    }
    input.close();
    return someObject;
  }


  private final static Object globalLock = "I'm a lock :)";


}
