package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * A simple entity linker based on Angel and Val's Wikipedia dictionary.
 *
 * @author Gabor Angeli
 */
public class WikidictEntityLinker extends EntityLinker {
  // Factor out variables
  private static final Redwood.RedwoodChannels logger = Redwood.channels("EntLink");
  private static final String FIELD_WORD = "word";
  private static final String FIELD_ARTICLE = "article";
  private static final String FIELD_SCORE = "score";
  private static final Set<String> FIELDS_TO_LOAD = new HashSet<>(Arrays.asList(FIELD_ARTICLE, FIELD_SCORE));

  // Lucene variables
  public final IndexReader reader;
  public final IndexSearcher searcher;
  public final Analyzer analyzer;
  protected EntityLinker backoffLinker = new GaborsHackyBaseline();

  /**
   * A strict version of the WikiDict entity linker, which does not employ a backoff linker if
   * the entities are not found in the dictionary.
   */
  public static class Strict extends WikidictEntityLinker {
    public Strict(IndexReader reader) { super(reader); super.backoffLinker = null; }
    public Strict(File reader) throws IOException { super(reader); super.backoffLinker = null; }
    public Strict() throws IOException { super(); super.backoffLinker = null; }
  }


  /** Create a new linker from a Lucene index reader pointing to the wikidict */
  public WikidictEntityLinker(IndexReader reader) {
    this.reader = reader;
    this.searcher = new IndexSearcher(this.reader);
    this.analyzer = new StandardAnalyzer(Version.LUCENE_42);
  }

  /** Create a new linker from a Lucene index path pointing to the wikidict */
  public WikidictEntityLinker(File reader) throws IOException {
    this(DirectoryReader.open(FSDirectory.open(reader)));
  }

  @SuppressWarnings("UnusedDeclaration") /** used via reflection */
  public WikidictEntityLinker() throws IOException {
    this(Props.INDEX_WIKIDICT);
  }

  /** A cache of the most recent queries, to avoid filesystem calls when possible */
  private final WeakHashMap<String, Counter<String>> cache = new WeakHashMap<>();

  /**
   * Return a set of articles which this string form often links to.
   * @param entityName The entity to try to link.
   * @return A set of candidate Wikipedia articles that may be associated with this entity.
   */
  private Counter<String> articlesForEntity(String entityName) {
    // Check cache
    synchronized (cache) {
      Counter<String> cachedValue = cache.get(entityName);
      if (cachedValue != null) {
        return cachedValue;
      }
    }
    // Hit lucene
    try {
      Counter<String> matchScores = new ClassicCounter<String>();
      Query query = new TermQuery(new Term(FIELD_WORD, entityName));
      TopFieldDocs results = this.searcher.search(query, 100, Sort.RELEVANCE);
      for (ScoreDoc result : results.scoreDocs) {
        Document doc = searcher.doc(result.doc, FIELDS_TO_LOAD);  // NOTE: update fieldsLuceneIdToDoc if you need more fields
        double score = Double.parseDouble(doc.get(FIELD_SCORE));
        if (score == 0.00) { break; }
        try {
          matchScores.incrementCount(doc.get(FIELD_ARTICLE), score);
        } catch (Exception e) {
          logger.err(e);
        }
      }
      // Populate cache
      synchronized (cache) {
        cache.put(entityName, matchScores);
      }
      return matchScores;
    } catch (IOException e) {
      logger.err(e);
      return new ClassicCounter<>();
    }
  }

  /**
   * @see WikidictEntityLinker#articlesForEntity(String)
   */
  private Counter<String> articlesForEntity(EntityContext context) {
    // Try vanilla search
    String name = context.entity.name;
    Counter<String> articles = articlesForEntity(context.entity.name);
    // Trim corporate suffixes
    if (articles.size() == 0) { name = stripCorporateTitles(name); articles = articlesForEntity(name); }
    // Trim determiners
    if (articles.size() == 0) { name = stripDeterminers(name); articles = articlesForEntity(name); }
    // Return
    return articles;
  }

  /** {@inheritDoc } */
  @Override
  public Maybe<String> link(EntityContext context) {
    return Maybe.Nothing();
  }

  /** {@inheritDoc } */
  @SuppressWarnings("ConstantConditions")
  @Override
  protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
    // -- Sanity Checks --
    // Exact match
    if (entityOne.entity.equals(entityTwo.entity)) { return true; }
    // Acronym
    if (AcronymMatcher.isAcronym(entityOne.entity.name, entityTwo.tokens()) ||
        AcronymMatcher.isAcronym(entityTwo.entity.name, entityOne.tokens())) {
      return true;
    }
    // Entity type
    if (entityOne.entity.type != entityTwo.entity.type) {
      return false;
    }
    // Get NER tag
    NERTag type = entityOne.entity.type;
    assert entityTwo.entity.type == type;
    assert entityOne.entity.type == type;
    // Gender
    if (type == NERTag.PERSON) {
      String firstNameOne = entityOne.tokens()[0].toLowerCase();
      String firstNameTwo = entityTwo.tokens()[0].toLowerCase();
      if ((maleNamesLowerCase.containsKey(firstNameOne) && femaleNamesLowerCase.containsKey(firstNameTwo)) ||
          (femaleNamesLowerCase.containsKey(firstNameOne) && maleNamesLowerCase.containsKey(firstNameTwo))) {
        return false;
      }
    }

    // -- Link --
    Counter<String> articlesOne = articlesForEntity(entityOne);
    Counter<String> articlesTwo = articlesForEntity(entityTwo);
    if (articlesOne.size() != 0 && articlesTwo.size() != 0) {
      // Link with Wikipedia
      Counter<String> intersection = new ClassicCounter<String>();
      for (String key : articlesOne.keySet()) {
        if (articlesTwo.containsKey(key)) {
          double a = articlesOne.getCount(key);
          double b = articlesTwo.getCount(key);
          intersection.incrementCount(key, 2 * a * b / (a + b));
        }
      }
      if (type == NERTag.PERSON) {
        return intersection.size() >= 1 && Counters.max(intersection) >= 0.9;  // people are much more finicky
      } else {
        return intersection.size() >= 1 && Counters.max(intersection) >= 0.5;
      }
    } else {
      // Hard disallow different last names (this should have been caught by wikidict)
      if (type == NERTag.PERSON && entityOne.tokens().length == 2 && entityTwo.tokens().length == 2) {
        String lastNameOne = entityOne.tokens()[entityOne.tokens().length - 1].toLowerCase();
        String lastNameTwo = entityTwo.tokens()[entityTwo.tokens().length - 1].toLowerCase();
        if (Utils.levenshteinDistance(lastNameOne.toLowerCase(), lastNameTwo.toLowerCase()) > Math.min(lastNameOne.length(), lastNameTwo.length()) / 4) {
          return false;
        }
      }
      // Backoff to more general linker
      //logger.log(backoffLinker+" is the backoff linker");
      return backoffLinker != null && backoffLinker.sameEntityWithoutLinking(entityOne, entityTwo);
    }
  }

  @Override
  protected void printJustification(EntityContext entityOne, EntityContext entityTwo) { /* noop */ }
}
