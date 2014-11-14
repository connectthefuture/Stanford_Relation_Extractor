package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.kbp.slotfilling.ir.index.KryoAnnotationSerializer;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.FileBackedCache;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * A test to complement the FileBackedCacheTest, but with the Kryo framework plugged in.
 *
 * @author Gabor Angeli
 */
public class KryoFileBackedCacheTest {

  private FileBackedCache<String, Annotation> cache;
  private StanfordCoreNLP pipeline;

  @Before
  public void setUp() {
    try {
      // Cache
      File cacheDir = File.createTempFile("cache", ".dir");
      assertTrue(cacheDir.delete());
      KryoAnnotationSerializer serializer = new KryoAnnotationSerializer(true, true);
      cache = serializer.createCache(cacheDir, 2, true);
      File[] files = cacheDir.listFiles();
      if (files != null) {
        assertEquals(0, files.length);
      } else {
        assertTrue(false);
      }

      // Pipeline
      Properties props = new Properties();
      props.setProperty("annotators", "tokenize,cleanxml");
      pipeline = new StanfordCoreNLP(props);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @After
  public  void tearDown() {
    if (cache.cacheDir.listFiles() != null) {
      File[] files = cache.cacheDir.listFiles();
      if (files != null) {
        for (File c : files) {
          assertTrue(c.delete());
        }
      }
      assertTrue(cache.cacheDir.delete());
    }
    pipeline = null;
  }

  private Annotation annotate(String str) {
    Annotation ann = new Annotation(str);
    pipeline.annotate(ann);
    return ann;
  }

  @Test
  public void testWrite() {
    cache.put("this is a test sentence. OK, maybe it's two sentences.",
        annotate("this is a test sentence. OK, maybe it's two sentences."));
    assertEquals(1, cache.size());
  }

  @Test
  public void testWriteMultiple() {
    cache.put("this is a test sentence. OK, maybe it's two sentences.",
        annotate("this is a test sentence. OK, maybe it's two sentences."));
    cache.put("Let's put another sentence in the cache. This is sentence 2.",
        annotate("Let's put another sentence in the cache. This is sentence 2."));
    cache.put("Let's put another sentence in the cache. This is sentence 3.",
        annotate("Let's put another sentence in the cache. This is sentence 3."));
    cache.put("Let's put another sentence in the cache. This is sentence 4.",
        annotate("Let's put another sentence in the cache. This is sentence 4."));
    cache.put("Let's put another sentence in the cache. This is sentence 5.",
        annotate("Let's put another sentence in the cache. This is sentence 5."));
    assertEquals(5, cache.size());
  }

  @Test
  public void testRead() {
    String sentence = "this is a test sentence. OK, maybe it's two sentences.";
    cache.put(sentence, annotate(sentence));
    assertEquals(1, cache.size());
    assertTrue(cache.containsKey(sentence));
    assertEquals(sentence, cache.get(sentence).get(CoreAnnotations.TextAnnotation.class));
  }

  @Test
  public void testReadMultiple() {
    String sentence = "this is a test sentence. OK, maybe it's two sentences.";
    String sentence2 = "this is a test sentence. Well, no, it's another sentence actually. For kicks, let's make it three sentences!.";
    cache.put(sentence, annotate(sentence));
    cache.put(sentence2, annotate(sentence2));
    assertEquals(2, cache.size());
    assertTrue(cache.containsKey(sentence));
    assertTrue(cache.containsKey(sentence2));
    assertEquals(sentence, cache.get(sentence).get(CoreAnnotations.TextAnnotation.class));
    assertEquals(sentence2, cache.get(sentence2).get(CoreAnnotations.TextAnnotation.class));
  }

  @Test
  public void readFromDisk() {
    String sentence = "this is a test sentence. OK, maybe it's two sentences.";
    String sentence2 = "this is a test sentence. Well, no, it's another sentence actually. For kicks, let's make it three sentences!.";
    cache.put(sentence, annotate(sentence));
    cache.put(sentence2, annotate(sentence2));
    assertEquals(2, cache.sizeInMemory());
    assertTrue(cache.removeFromMemory(sentence));
    assertEquals(1, cache.sizeInMemory());
    assertTrue(cache.removeFromMemory(sentence2));
    assertEquals(0, cache.sizeInMemory());
    assertEquals(2, cache.size());
    assertTrue(cache.containsKey(sentence));
    assertTrue(cache.containsKey(sentence2));
    assertEquals(sentence, cache.get(sentence).get(CoreAnnotations.TextAnnotation.class));
    assertEquals(sentence2, cache.get(sentence2).get(CoreAnnotations.TextAnnotation.class));
  }

}
