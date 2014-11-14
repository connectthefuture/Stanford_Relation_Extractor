package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.KBPNew;
import edu.stanford.nlp.kbp.common.KBPSlotFill;
import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.GrammaticalRelation;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * A test for the various
 * {@link edu.stanford.nlp.kbp.slotfilling.process.FeatureProvider}s implemented in
 * {@link edu.stanford.nlp.kbp.slotfilling.process.FeatureProviders}.
 *
 * @author Gabor Angeli
 */
public class FeatureProvidersTest {

  private CoreLabel lbl(int indexFromZero, String word, String lemma, String pos, String ner, SemanticGraph dependencies) {
    CoreLabel lbl = new CoreLabel();
    lbl.setWord(word);
    lbl.setOriginalText(word);
    lbl.setLemma(lemma);
    lbl.setTag(pos);
    lbl.setNER(ner);
    lbl.setDocID("docid");
    lbl.setSentIndex(0);
    lbl.setIndex(indexFromZero + 1);
    IndexedWord indexedWord = new IndexedWord(lbl);
    dependencies.addVertex(indexedWord);
    return lbl;
  }

  private void edge(SemanticGraph dependencies, int start, int end, String lbl) {
    dependencies.addEdge(
        start == 0 ? dependencies.getFirstRoot() : dependencies.getNodeByIndex(start),
        dependencies.getNodeByIndex(end),
        GrammaticalRelation.valueOf(lbl), 1.0, false);
  }

  /**
   * George Bush was born in Texas.
   */
  private Featurizable georgeBushWasBornInTexas() {
    final SemanticGraph dependencies = new SemanticGraph();
    dependencies.setRoot(new IndexedWord("ROOT", 0, 0));
    // Create tokens
    List<CoreLabel> tokens = new ArrayList<CoreLabel>() {{
      add(lbl(0, "George", "George", "NNP", NERTag.PERSON.name, dependencies));
      add(lbl(1, "Bush", "Bush", "NNP", NERTag.PERSON.name, dependencies));
      add(lbl(2, "was", "be", "VB", "O", dependencies));
      add(lbl(3, "born", "bear", "VB", "O", dependencies));
      add(lbl(4, "in", "in", "P", "O", dependencies));
      add(lbl(5, "Texas", "Texas", "NNP", NERTag.STATE_OR_PROVINCE.name, dependencies));
      add(lbl(6, ".", ".", ".", "O", dependencies));
    }};
    // Create dependency graph
    edge(dependencies, 0, 4, "root");
    edge(dependencies, 2, 1, "nn");
    edge(dependencies, 4, 2, "nsubjpass");
    edge(dependencies, 4, 3, "auxpass");
    edge(dependencies, 4, 5, "prep");
    edge(dependencies, 5, 6, "pobj");
    // Create spans
    Span entity = new Span(0, 2);
    Span slotValue = new Span(5, 6);
    // Create OpenIE slot fill
    KBPSlotFill fill = KBPNew.entName("George Bush").entType(NERTag.PERSON).slotValue("Texas").slotType(NERTag.STATE_OR_PROVINCE).rel("born in").KBPSlotFill();
    // Return
    return new Featurizable(entity, slotValue, tokens, dependencies, Collections.singleton(fill));
  }

  /**
   * Solange Knowles, singer Beyonce's younger sister, is also a singer.
   */
  private Featurizable solangeKnowles() {
    final SemanticGraph dependencies = new SemanticGraph();
    dependencies.setRoot(new IndexedWord("ROOT", 0, 0));
    // Create tokens
    List<CoreLabel> tokens = new ArrayList<CoreLabel>() {{
      add(lbl(0, "Solange", "Solange", "NNP", NERTag.PERSON.name, dependencies));
      add(lbl(1, "Knowles", "Knowles", "NNP", NERTag.PERSON.name, dependencies));
      add(lbl(2, ",", ",", ".", "O", dependencies));
      add(lbl(3, "singer", "singer", "NN", NERTag.TITLE.name, dependencies));
      add(lbl(4, "Beyonce", "Beyonce", "NNP", NERTag.PERSON.name, dependencies));
      add(lbl(5, "'s", "'s", "POS", "O", dependencies));
      add(lbl(6, "younger", "younger", "JJR", "O", dependencies));
      add(lbl(7, "sister", "sister", "NN", "O", dependencies));
      add(lbl(8, ",", ",", ".", "O", dependencies));
      add(lbl(9, "is", "be", "VBZ", "O", dependencies));
      add(lbl(10, "also", "also", "RB", "O", dependencies));
      add(lbl(11, "a", "a", "DT", "O", dependencies));
      add(lbl(12, "singer", "singer", "NN",NERTag.TITLE.name, dependencies));
      add(lbl(13, ".", ".", ".", "O", dependencies));
    }};
    // Create dependency graph
    edge(dependencies, 0, 13, "root");
    edge(dependencies, 2, 1, "nn");
    edge(dependencies, 13, 2, "nsubj");
    edge(dependencies, 2, 4, "appos");
    edge(dependencies, 8, 5, "poss");
    edge(dependencies, 5, 6, "possessive");
    edge(dependencies, 8, 7, "amod");
    edge(dependencies, 4, 8, "dep");
    edge(dependencies, 13, 10, "cop");
    edge(dependencies, 13, 11, "advmod");
    edge(dependencies, 13, 12, "det");
    // Create spans
    Span entity = new Span(0, 2);
    Span slotValue = new Span(4, 5);
    // Create slot fill
    KBPSlotFill fill = KBPNew.entName("Beyonce Knowles").entType(NERTag.PERSON).slotValue("Solange").slotType(NERTag.PERSON).rel("sister").KBPSlotFill();
    // Return
    return new Featurizable(entity, slotValue, tokens, dependencies, Collections.singleton(fill));
  }

  @Test
  public void lexBetweenWordUnigram() {
    assertEquals(
        new HashSet<String>() {{
          add("was");
          add("born");
          add("in");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenWordUnigram("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add(",");
          add("singer");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenWordUnigram("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void lexBetweenWordBigram() {
    assertEquals(
        new HashSet<String>() {{
          add("^_was");
          add("was_born");
          add("born_in");
          add("in_$");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenWordBigram("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add("^_,");
          add(",_singer");
          add("singer_$");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenWordBigram("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void lexBetweenLemmaUnigram() {
    assertEquals(
        new HashSet<String>() {{
          add("be");
          add("bear");
          add("in");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenLemmaUnigram("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add(",");
          add("singer");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenLemmaUnigram("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void lexBetweenLemmaBigram() {
    assertEquals(
        new HashSet<String>() {{
          add("^_be");
          add("be_bear");
          add("bear_in");
          add("in_$");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenLemmaBigram("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add("^_,");
          add(",_singer");
          add("singer_$");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenLemmaBigram("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void lexBetweenNER() {
    assertEquals(
        new HashSet<String>() {{
        }},
        new HashSet<>(new FeatureProviders.LexBetweenNER("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add(NERTag.TITLE.name);
        }},
        new HashSet<>(new FeatureProviders.LexBetweenNER("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void lexBetweenPunctuation() {
    assertEquals(
        new HashSet<String>() {{
        }},
        new HashSet<>(new FeatureProviders.LexBetweenPunctuation("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add("");
        }},
        new HashSet<>(new FeatureProviders.LexBetweenPunctuation("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void depBetweenWordUnigram() {
    assertEquals(
        new HashSet<String>() {{
          add("born");
          add("in");
        }},
        new HashSet<>(new FeatureProviders.DepBetweenWordUnigram("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add("singer");
          add("sister");
        }},
        new HashSet<>(new FeatureProviders.DepBetweenWordUnigram("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void depBetweenWordLemma() {
    assertEquals(
        new HashSet<String>() {{
          add("bear");
          add("in");
        }},
        new HashSet<>(new FeatureProviders.DepBetweenLemmaUnigram("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add("singer");
          add("sister");
        }},
        new HashSet<>(new FeatureProviders.DepBetweenLemmaUnigram("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void depBetweenNER() {
    assertEquals(
        new HashSet<String>() {{
        }},
        new HashSet<>(new FeatureProviders.DepBetweenNER("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add(NERTag.TITLE.name);
        }},
        new HashSet<>(new FeatureProviders.DepBetweenNER("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void nerSignatureEntity() {
    assertEquals(
        new HashSet<String>() {{
          add("PERSON");
        }},
        new HashSet<>(new FeatureProviders.NERSignatureEntity("prefix").featureValues(georgeBushWasBornInTexas())));
  }

  @Test
  public void nerSignatureSlotValue() {
    assertEquals(
        new HashSet<String>() {{
          add("STATE_OR_PROVINCE");
        }},
        new HashSet<>(new FeatureProviders.NERSignatureSlotValue("prefix").featureValues(georgeBushWasBornInTexas())));
  }

  @Test
  public void nerSignature() {
    assertEquals(
        new HashSet<String>() {{
          add("PERSON_STATE_OR_PROVINCE");
        }},
        new HashSet<>(new FeatureProviders.NERSignature("prefix").featureValues(georgeBushWasBornInTexas())));
  }

  @SuppressWarnings({"UnusedDeclaration", "MismatchedQueryAndUpdateOfCollection"})
  @Test
  public void openIESimplePatternsCheckSubjObjSemgrex() {
    final SemanticGraph dependencies = new SemanticGraph();
    dependencies.setRoot(new IndexedWord("ROOT", 0, 0));
    // Create tokens
    List<CoreLabel> tokens = new ArrayList<CoreLabel>() {{
      add(lbl(0, "Dogs", "Dog", "NN", NERTag.PERSON.name, dependencies));
      add(lbl(1, "chase", "chase", "VB", "O", dependencies));
      add(lbl(2, "after", "after", "IN", "O", dependencies));
      add(lbl(3, "cats", "cat", "NN", NERTag.PERSON.name, dependencies));
      add(lbl(4, ".", ".", ".", "O", dependencies));
    }};
    // Create dependency graph
    edge(dependencies, 0, 2, "root");
    edge(dependencies, 2, 1, "nsubj");
    edge(dependencies, 2, 3, "prep");
    edge(dependencies, 3, 4, "pobj");
    // Check pattern
    SemgrexMatcher matcher = new FeatureProviders.OpenIESimplePatterns("prefix").subj_obj.matcher(dependencies);
    assertTrue(matcher.find());
    assertEquals("chase", matcher.getNode("rel").word());
    assertEquals("Dogs", matcher.getNode("subj").word());
    assertEquals("cats", matcher.getNode("obj").word());
  }

  @Test
  public void openIESimplePatterns() {
    assertEquals(
        new HashSet<String>() {{
          add("subj<-born->obj");
        }},
        new HashSet<>(new FeatureProviders.OpenIESimplePatterns("prefix").featureValues(georgeBushWasBornInTexas())));
  }

  @Test
  public void openIERelation() {
    assertEquals(
        new HashSet<String>() {{
          add("subj<-born in->obj");
        }},
        new HashSet<>(new FeatureProviders.OpenIERelation("prefix").featureValues(georgeBushWasBornInTexas())));
    assertEquals(
        new HashSet<String>() {{
          add("subj<-sister->obj");
        }},
        new HashSet<>(new FeatureProviders.OpenIERelation("prefix").featureValues(solangeKnowles())));
  }

  @Test
  public void allFeaturesFire() {
    Featurizable mention = georgeBushWasBornInTexas();
    Counter<String> counts = new ClassicCounter<>();
    // Featurize
    for (Feature feat : Feature.values()) {
      feat.provider.apply(mention, counts);
    }
    // Check
    // (set of features which shouldn't fire)
    Set<Feature> shouldntFire = new HashSet<Feature>(){{
      add(Feature.LEX_BETWEEN_NER);
      add(Feature.LEX_BETWEEN_PUNCTUATION);
      add(Feature.DEP_BETWEEN_NER);
    }};
    // (check if the other features fire)
    for (Feature feat : Feature.values()) {
      if (shouldntFire.contains(feat)) { continue; }
      int numFound = 0;
      for (String key : counts.keySet()) {
        if (key.startsWith(feat.provider.prefix)) {
          numFound += 1;
        }
      }
      assertTrue("Feature didn't fire: " + feat.name(), numFound > 0);
    }
  }
}
