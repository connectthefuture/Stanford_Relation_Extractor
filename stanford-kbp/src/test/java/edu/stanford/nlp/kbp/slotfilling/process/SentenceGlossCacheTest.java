package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.CoreMapUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.StringUtils;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import static junit.framework.Assert.*;

/**
 * A test to make sure that sentence glossing is robust to common sentence variations.
 *
 * @author Gabor Angeli
 */
public class SentenceGlossCacheTest {

  public static StanfordCoreNLP pipeline = new StanfordCoreNLP(new Properties(){{
    setProperty("annotators", "tokenize, ssplit");
  }});
  public static List<String> sentence = Arrays.asList("Julie", "named", "Julie", "was", "born", "in", "Canada");
  public static Span entitySpan = new Span(0, 1);
  public static Span slotSpan = new Span(6, 7);
  /** Computed from: http://www.xorbin.com/tools/sha256-hash-calculator */
  public static String expectedKey = "200acf98ab1461ca7e98d7e495c23c64b573f845d2c5de850208afab607fb35b:0-1:6-7";

  @Test
  public void testStringCase() {
    String hexKey = CoreMapUtils.getSentenceGlossKey(sentence.toArray(new String[sentence.size()]), entitySpan, slotSpan);
    assertEquals(expectedKey, hexKey);
  }

  @Test
  public void testCoreNLPAnnotatedCase() {
    Annotation ann = new Annotation(StringUtils.join(sentence, " "));
    pipeline.annotate(ann);
    String hexKey = CoreMapUtils.getSentenceGlossKey(
        ann.get(CoreAnnotations.SentencesAnnotation.class).get(0).get(CoreAnnotations.TokensAnnotation.class), entitySpan, slotSpan);
    assertEquals(expectedKey, hexKey);
  }

  @Test
  public void testCoreNLPAnnotatedCaseStrangeWhitespace() {
    Annotation ann = new Annotation("\n" + StringUtils.join(sentence, "\t \n\n") + "\t  ");
    pipeline.annotate(ann);
    String hexKey = CoreMapUtils.getSentenceGlossKey(
        ann.get(CoreAnnotations.SentencesAnnotation.class).get(0).get(CoreAnnotations.TokensAnnotation.class), entitySpan, slotSpan);
    assertEquals(expectedKey, hexKey);
  }

  @Test
  public void testCaseSensitive() {
    String hexKey = CoreMapUtils.getSentenceGlossKey(StringUtils.join(sentence).toLowerCase().split(" "), entitySpan, slotSpan);
    assertFalse(expectedKey.equals(hexKey));
  }

  @Test
  public void testMultipleEntitiesWithSameName() {
    String hexKey = CoreMapUtils.getSentenceGlossKey(sentence.toArray(new String[sentence.size()]), new Span(2, 3), slotSpan);
    assertFalse(expectedKey.equals(hexKey));
  }

}
