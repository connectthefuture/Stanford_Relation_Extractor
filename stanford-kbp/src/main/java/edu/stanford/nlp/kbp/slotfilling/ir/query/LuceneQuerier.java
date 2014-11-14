package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KBPField;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.search.spans.SpanMultiTermQueryWrapper;
import org.apache.lucene.search.spans.SpanNearQuery;
import org.apache.lucene.search.spans.SpanQuery;
import org.apache.lucene.search.spans.SpanTermQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.Counter;
import org.apache.lucene.util.Version;

import static org.apache.lucene.search.BooleanClause.Occur.*;
import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * The base class for querying Lucene for documents.
 * This class has a large number of utility methods, falling into two
 * categories: helpers for subclasses (marked as protected),
 * and helpers for callers (marked as public).
 *
 * Subclasses of this class should implement the queryImplementation().
 *
 * Some use cases
 *   Query for training: entity, relation, slot fill
 *   Query for provenances: entity, relation, slot fill
 *   Query for candidate sentences during testing: entity
 *
 * @author Gabor Angeli
 */
public abstract class LuceneQuerier implements Querier {
  //
  //  CONSTRUCTOR AND VARIABLES
  //
  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Lucene");

  // Directory to identify this querier with
  public final Maybe<File> indexDirectory;

  // Lucene variables
  protected final IndexReader reader;
  protected final IndexSearcher searcher;
  protected final Analyzer analyzer;
  protected final LuceneDocumentReader docReader;
  protected final Set<String> knownFields; // Known fields associated with this index

  // Utilities
  // Tokenizer to get consistent tokenization for indexing and querying
  protected final Annotator tokenizer;
  // Pipeline to re-annotate saved away coremap
  // Use for fixes to annotation after indexing
  // Once fixes are confirmed to be good, annotations should be reindexed so we don't have to re-annotate here
  // Do re-annotation at the stage when the document is fetched so that querySentences will also
  //   take advantage of re-annotation (e.g. important for changes in coref)
  protected AnnotationPipeline reannotatePipeline;

  protected LuceneQuerier(IndexReader reader) {
    this.reader = reader;
    this.searcher = new IndexSearcher(this.reader);
    this.analyzer = new StandardAnalyzer(Version.LUCENE_42);

    if (reader instanceof DirectoryReader && ((DirectoryReader) reader).directory() instanceof FSDirectory) {
      indexDirectory = Maybe.Just(((FSDirectory) ((DirectoryReader) reader).directory()).getDirectory());
    } else {
      indexDirectory = Maybe.Nothing();
    }

    docReader = LuceneDocumentReader.KBPIndexVersion.fromReader(reader).reader;
    try {
      knownFields = LuceneUtils.getStoredFields(reader);
      logger.debug("got known fields: " + knownFields);
    } catch (IOException ex) {
      throw new RuntimeException("Cannot get any fields from index: check index is okay " + indexDirectory, ex);
    }
    // TODO: set tokenizer options?
    tokenizer = new TokenizerAnnotator(false);
  }

  public void setReannotatePipeline(AnnotationPipeline pipeline) {
    this.reannotatePipeline = pipeline;
  }
  //
  //  PUBLIC FACING METHODS
  //

  /** Clean up the files associated with this querier */
  @Override
  public void close() {
    try {
      this.reader.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /** This is the major method to override, and the heart of LuceneQuerier */
  protected abstract IterableIterator<Pair<Integer, Double>> queryImplementation(String entityName,
                                                           Maybe<NERTag> entityType,
                                                           Maybe<String> relation,
                                                           Maybe<String> slotValue,
                                                           Maybe<NERTag> slotValueType,
                                                           Maybe<Integer> maxDocuments) throws IOException;

  /**
   * Get the total number of documents containing a set of search terms.
   * The semantics of the search is always that every term in the collection must occur as an exact phrase.
   * @param terms The query phrases to search for.
   * @return The number of results in the index.
   */
  public int queryHitCount(Collection<String> terms) {
    BooleanQuery query = new BooleanQuery();
    for (String term : terms) {
      PhraseQuery pq = new PhraseQuery();
      pq.setSlop(2);
      for (String token : term.split("\\s+")) {
        pq.add(new Term(KBPField.TEXT_WORD.fieldName(), token));
      }
      query.add(new BooleanClause(pq, MUST));
    }
    try {
      return searcher.search(query, Integer.MAX_VALUE).totalHits;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /** Query only a document ID; this doesn't require reading in the CoreMap */
  public IterableIterator<Pair<String, Double>> queryDocId(String entityName,
                                                           Maybe<NERTag> entityType,
                                                           Maybe<String> relation,
                                                           Maybe<String> slotValue,
                                                           Maybe<NERTag> slotValueType,
                                                           Maybe<Integer> maxDocuments) {
    try {
      startTrack("queryDocId");
      logger.log("queryDocId: entity=" + entityName + "(" + entityType +  ")"
              + ", relation=" + relation + ", slotValue=" + slotValue + "(" + slotValueType + ")");
      return CollectionUtils.mapIgnoreNull(queryImplementation(entityName, entityType, relation, slotValue, slotValueType, maxDocuments),
          docPair -> {
            try {

              // --- CODE BEGINS HERE ---
              // -- Get Document ID
              int luceneId = docPair.first;
              Document doc = searcher.doc(luceneId, new HashSet<>(
                      Arrays.asList(new String[]{ "date", KBPField.DOCID.fieldName() })));
              String docid = docReader.getDocid(doc);
              // Return
              return Pair.makePair(docid == null ? "<no id>" : docid.trim(), docPair.second);
              // --- CODE ENDS HERE ---

            } catch (IOException e) {
              logger.err(e);
              return null;
            }
          }
      );
    } catch (IOException e) {
      logger.err(e);
      return new IterableIterator<>(new LinkedList<Pair<String,Double>>().iterator());
    } finally {
      endTrack("queryDocId");
    }
  }

  /** Query documents (as annotations) */
  public IterableIterator<Pair<Annotation, Double>> queryDocument(KBPEntity entity,
                                                       Maybe<KBPEntity> slotValue,
                                                       Set<String> docidsToForce,
                                                       Maybe<Integer> maxDocuments) {
    try {
      startTrack("Query Document");
      logger.log("queryDocuments: entity=" + entity.name + "(" + entity.type +  ")"
              + ", slotValue=" + slotValue);
      Maybe<String> slotValueName = slotValue.isDefined() ? Maybe.Just(slotValue.get().name) : Maybe.<String>Nothing();
      Maybe<NERTag> slotValueType = slotValue.isDefined() ? Maybe.Just(slotValue.get().type) : Maybe.<NERTag>Nothing();
      IterableIterator<Pair<Integer, Double>> docs = queryImplementation(entity.name, Maybe.Just(entity.type), Maybe.<String>Nothing(), slotValueName, slotValueType, maxDocuments);
      if (!docidsToForce.isEmpty()) {
        // Add in any forced documents
        ArrayList<Pair<Integer, Double>> docsWithForced = new ArrayList<>();
        Set<Integer> forcedIds = new HashSet<>();
        for (String docid : docidsToForce) {
          TopDocs results = this.searcher.search(new TermQuery(new Term(KBPField.DOCID.fieldName(), docid)), 1);
          if (results.scoreDocs.length > 0) {
            forcedIds.add(results.scoreDocs[0].doc);
            docsWithForced.add(Pair.makePair(results.scoreDocs[0].doc, 1.0));
          }
        }
        for (Pair<Integer, Double> entry : docs) { if (!forcedIds.contains(entry.first)) { docsWithForced.add(entry); } }
        docs = new IterableIterator<>(docsWithForced.iterator());
      }
      startTrack("Read Documents");
      return CollectionUtils.mapIgnoreNull(docs, fetchDocument);
    } catch (IOException e) {
      logger.err(e);
      return new IterableIterator<>(new LinkedList<Pair<Annotation,Double>>().iterator());
    } finally {
      endTrackIfOpen("Read Documents");
      endTrack("Query Document");
    }
  }

  /** Query sentences (as CoreMaps) */
  @Override
  public IterableIterator<Pair<CoreMap, Double>> querySentences(KBPEntity entity,
        Maybe<KBPEntity> slotValue,
        Maybe<String> relationName,
        Set<String> docidsToForce,
        Maybe<Integer> maxDocuments) {
    final AtomicInteger docsSeen = new AtomicInteger(0);
    startTrack("Query Sentences");
    try {
      return CollectionUtils.flatMapIgnoreNull(queryDocument(entity, slotValue, docidsToForce, maxDocuments),
          fetchSentences(entity, slotValue, maxDocuments, docsSeen));
    } finally {
      endTrack("Query Sentences");
    }
  }

  /**
   * {@inheritDoc}
   * <p>Note that a dummy PostIRAnnotator is run on this document once it has left this method. </p>
   */
  @Override
  public Stream<Annotation> slurp(int maxDocuments) {
    // Variables we'll need
    final Bits liveDocs = MultiFields.getLiveDocs(reader);

    final PostIRAnnotator dummyPostIR = new PostIRAnnotator(KBPNew.entName("__NO ENTITY__").entType(NERTag.PERSON).KBPOfficialEntity());
    if(maxDocuments > reader.maxDoc())
      maxDocuments  = reader.maxDoc();
    // Create stream
    Stream<Integer> idStream = StreamSupport.stream(CollectionUtils.seqIter(maxDocuments).spliterator(), true);
    return idStream.map(luceneDocId -> {
      if (liveDocs == null || liveDocs.get(luceneDocId)) {
        Annotation doc = fetchDocument.apply(Pair.makePair(luceneDocId, 1.0)).first;
        dummyPostIR.annotate(doc);
        return doc;
      } else {
        return null;
      }
    }).filter(ann -> ann != null);
  }

  /**
   * Fetch a given document from this index, if it exists.
   * @param docId The id of the document to fetch.
   * @return The document as a CoreNLP {@link Annotation}, or {@link edu.stanford.nlp.kbp.common.Maybe#Nothing} if the document does not exist.
   * @throws IOException From the underlying Lucene implementation.
   */
  public Maybe<Annotation> fetchDocument(String docId) throws IOException {
    TopDocs results = this.searcher.search(new TermQuery(new Term(KBPField.DOCID.fieldName(), docId)), 1);
    if (results.scoreDocs.length == 0) {
      return Maybe.Nothing();
    }
    return Maybe.Just(fetchDocument.apply(Pair.makePair(results.scoreDocs[0].doc, 1.0)).first);
  }

  public IterableIterator<Pair<Annotation, Double>> queryKeywords(Collection<String> words, Maybe<Integer> maxDocuments){
    BooleanQuery q = new BooleanQuery();
    for (String w:words) {
      q.add( qrewrite(w, LuceneQuerierParams.strict().withCaseSensitive(false)), BooleanClause.Occur.SHOULD );
    }
    TopDocs docs = null;
    try {
      docs = queryWithTimeout(q, maxDocuments, Props.INDEX_LUCENE_TIMEOUTMS);
      logger.log("query: " + q + " got: " + docs.scoreDocs.length + "/" + docs.totalHits);
      return CollectionUtils.mapIgnoreNull(asIterator(docs), fetchDocument);
    }catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }


  //
  //  UTILITIES FOR SUBCLASSES
  //

  /**
   * <p>Run the specified query with a timeout in place. This is recommended to avoid runaway query times,
   * especially at datum caching time.</p>
   *
   * <p>If the query time is exceeded, the results which are available at the timeout are returned; though this may
   * be an imcomplete list.</p>
   *
   * @param query The query to run
   * @param maxDocuments The maximum number of documents to query (Maybe.Nothing() for no limit)
   * @param timeoutInMS The timeout, in milliseconds. This is guaranteed to be in real time, but limited by
   *                    whether the timer thread gets focus; thus, the query will run for a maximum of
   *                    at least this amount of time, and hopefully not much more.
   * @return The result of the query
   * @throws IOException Passed from searcher.search()
   */
  protected TopDocs queryWithTimeout(Query query, Maybe<Integer> maxDocuments, int timeoutInMS) throws IOException {
    // -- Setup Timeout
    final Counter ticks = Counter.newCounter();
    TopScoreDocCollector scoreCollector =  TopScoreDocCollector.create(maxDocuments.getOrElse(Integer.MAX_VALUE), true);
    TimeLimitingCollector timedCollector = new TimeLimitingCollector(scoreCollector, ticks, timeoutInMS);
    final long searchStart = System.currentTimeMillis();
    if (Props.INDEX_LUCENE_TIMEOUTMS < Integer.MAX_VALUE) {
      Thread ticker = new Thread() {
        @Override public void run() {
          while (ticks.addAndGet(System.currentTimeMillis() - searchStart) < Props.INDEX_LUCENE_TIMEOUTMS) {
            //noinspection EmptyCatchBlock
            try { Thread.sleep(10); } catch (InterruptedException e) { }
            ticks.addAndGet(System.currentTimeMillis() - searchStart);
          }
        }
      };
      ticker.setDaemon(true);
      ticker.start();
    }

    // -- Run Query
    try {
      this.searcher.search(query, Props.INDEX_LUCENE_TIMEOUTMS < Integer.MAX_VALUE ? timedCollector : scoreCollector);
    } catch (TimeLimitingCollector.TimeExceededException e) {
      logger.warn("query timed out!");
    }

    return scoreCollector.topDocs();
  }

  private static boolean acceptWord(String word) {
    // Filter empty words
    if (word.isEmpty()) return false;
    // Filter stop words
    if (org.apache.lucene.analysis.standard.StandardAnalyzer.STOP_WORDS_SET.contains(word.toLowerCase())) { return false; }
    // Filter symbols
    if (word.length() == 1) {
      char c = word.charAt(0);
      if (c < 47) return false;
      if (c > 57 && c < 65) return false;
      if (c > 90 && c < 97) return false;
      if (c > 122) return false;
    }
    return true;
  }

  private static String[] wordRewrite(String word, boolean split) {
    // don't rewrite URLs
    Matcher url = REGEX_URL.matcher(word);
    if (url.matches()) { return new String[]{ word }; }
    // lucene doesn't like certain characters
    String[] words = (split)? word.split("\\.|\\,|\\-"):new String[]{word};
    // unescape xml
    for (int i = 0; i < words.length; ++i) {
      String w = words[i];
      if (w.trim().length() == 0) {
        words[i] = null;
      } else if (w.charAt(0) == '&' && w.charAt(w.length() - 1) == ';') {
        words[i] = StringEscapeUtils.unescapeXml(w);
      }
    }
    // remove null words
    int notNullCount = 0;
    for (String w : words) { notNullCount += (w == null ? 0 : 1); }
    if (notNullCount == words.length) { return words; }
    String[] cleanWords = new String[notNullCount];
    int cleanI = 0;
    for (String w : words) {
      if (w != null) { cleanWords[cleanI] = w; cleanI += 1; }
    }
    // return
    return cleanWords;
  }

  protected Query qrewrite(String[] words,
                           LuceneQuerierParams params) {
    // Very simple straw man policy to show improvement against
    if (Props.INDEX_MODE == Props.QueryMode.DUMB) {
      PhraseQuery query = new PhraseQuery();
      for (String word : words) {
        query.add(new Term(KBPField.TEXT_WORD.fieldName(), word));
      }
      return query;
    }

    // Filter words
    List<Pair<Term, Class>> terms = new ArrayList<>();
    boolean queryPreTokenizedText = (params.queryPreTokenizedText && knownFields.contains(KBPField.TEXT_WORD.fieldName()));
    for (String word : words) {
      // Case sensitivity
      word = params.caseSensitive ? word.trim() : word.trim().toLowerCase();
      // Filter invalid (e.g., stop) words
      if (!params.phraseSemantics.equals(LuceneQuerierParams.PhraseSemantics.PHRASE))
        if (!acceptWord(word)) { continue; }
      // Query type
      Class queryType = TermQuery.class;
      if (Props.INDEX_LUCENE_ABBREVIATIONS_DO && word.endsWith(".") &&
          ( (word.length() == 2 && Character.isUpperCase(word.charAt(0)) && Character.isAlphabetic(word.charAt(0))) ||
            (word.length() > 2 && Character.isUpperCase(word.charAt(0)) && Character.isLowerCase(word.charAt(word.length() - 2)) &&
              Character.isAlphabetic(word.charAt(0)) && Character.isAlphabetic(word.charAt(word.length() - 2)))
          )
         ) {
        // Expand acronyms / abbreviations
        logger.log("wildcard query for abbreviation: " + word);
        word  = word.substring(0, word.length() - 1) + "*";
        queryType = WildcardQuery.class;
      }
      // Rewrite word - query tokenization need to match indexing tokenization
      String[] tokenizedWord = wordRewrite(word, !queryPreTokenizedText);
      // Add term
      for (String w : tokenizedWord) {
        if (queryPreTokenizedText) {
          if (params.caseSensitive) {
            terms.add(Pair.makePair(new Term(KBPField.TEXT_WORD.fieldName(), w), queryType));
          } else {
            terms.add(Pair.makePair(new Term(KBPField.TEXT_WORD_NORM.fieldName(), w), queryType));
          }
        } else {
          terms.add(Pair.makePair(new Term(KBPField.TEXT.fieldName(), w), queryType));
        }
      }
    }

    // Construct query parts
    Query[] queryParts = new Query[terms.size()];
    for (int i = 0; i < terms.size(); ++i) {
      Term term = terms.get(i).first;
      queryParts[i] = MetaClass.create(terms.get(i).second).createInstance(term);
    }

    // Construct Query
    Query query;
    boolean orderedSpan = false;
    switch (params.phraseSemantics) {
      case PHRASE:
        query = new PhraseQuery();
        for (Pair<Term, Class> term : terms) {
          if (term.second.equals(WildcardQuery.class)) { return qrewrite(words, params.withPhraseSemantics(LuceneQuerierParams.PhraseSemantics.SPAN_ORDERED)); }
          ((PhraseQuery) query).add(term.first);
        }
        break;
      case SPAN_ORDERED:
        orderedSpan = true; // falls through to below
      case SPAN_UNORDERED:
        List<SpanQuery> spanQueries = new ArrayList<>();
        for (Pair<Term, Class> term : terms) {
          if (term.second.equals(WildcardQuery.class)) {
            spanQueries.add(new SpanMultiTermQueryWrapper<>(new WildcardQuery(term.first)));
          } else {
            spanQueries.add(new SpanTermQuery(term.first));
          }
        }
        query = new SpanNearQuery(spanQueries.toArray( new SpanQuery[spanQueries.size()]), params.slop, orderedSpan);
        break;
      case UNIGRAMS_MUST:
        query = new BooleanQuery();
        for (Query component : queryParts) {
          ((BooleanQuery) query).add(component, MUST);
        }
        break;
      case UNIGRAMS_SHOULD:
        query = new BooleanQuery();
        for (Query component : queryParts) {
          ((BooleanQuery) query).add(component, SHOULD);
        }
        break;
      default:
        throw new IllegalStateException("Unknown query parameters setting: " + params.phraseSemantics);
    }

    // Return
    return query;
  }

  protected String[] tokenize(String phrase) {
    Annotation anno = new Annotation(phrase);
    tokenizer.annotate(anno);
    List<CoreLabel> tokens = anno.get(TokensAnnotation.class);
    List<String> words = new ArrayList<>();
    for (int i = 0; i < tokens.size(); i++) {
      if (i+1 < tokens.size() && ".".equals(tokens.get(i+1).word())) {
        words.add( tokens.get(i).originalText() + "." );
        i++;
      } else {
        words.add( tokens.get(i).originalText() );
      }
    }
    return words.toArray(new String[words.size()]);
  }

  protected Query qrewrite(String phrase, LuceneQuerierParams params) {
    String[] tokens = tokenize(phrase);
    return qrewrite(tokens, params);
  }

  protected static IterableIterator<Pair<Integer, Double>> asIterator(TopDocs results) {
    final ScoreDoc[] scores = results.scoreDocs;
    return new IterableIterator<>(new Iterator<Pair<Integer,Double>>() {
      int i = 0;
      @Override
      public boolean hasNext() {
        return i < scores.length;
      }
      @Override
      public Pair<Integer, Double> next() {
        if (i >= scores.length) {
          throw new NoSuchElementException();
        }
        ScoreDoc doc = scores[i];
        i += 1;
        Integer docId = doc.doc;
        Double docScore = (double) doc.score;
        return Pair.makePair(docId, docScore);
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException("Why on earth are you removing an element from a Lucene result?");
      }
    });
  }

  @Override
  public String toString() {
    return this.getClass().getSimpleName() + "(" + this.indexDirectory.getOrElse(new File("<unknown location>")).getPath() + ")";
  }


  private final Function<Pair<Integer, Double>, Pair<Annotation, Double>> fetchDocument =
      new Function<Pair<Integer, Double>, Pair<Annotation, Double>>() {
        @SuppressWarnings("ConstantConditions")
        @Override
        public Pair<Annotation, Double> apply(Pair<Integer, Double> docPair) {
          try {

            // --- CODE BEGINS HERE ---
            // -- Get Annotation
            int luceneId = docPair.first;

            Timing timing = null; // new Timing();

            // Fetch lucene document
            if (timing != null) timing.start();
            Document doc = searcher.doc(luceneId, fieldsLuceneIdToDoc);  // NOTE: update fieldsLuceneIdToDoc if you need more fields
            if (timing != null) timing.report("Fetch doc " + luceneId + " from " + indexDirectory);

            // Get document id
            Maybe<String> docId = Maybe.Nothing();
            String idOrNull = docReader.getDocid(doc);
            if (idOrNull != null) { docId = Maybe.Just(idOrNull.trim()); }
            if (!docId.isDefined()) { logger.warn("could not find docid for document!"); }

            // Load coremap
            if (timing != null) timing.start();
            Maybe<Annotation> coremapDocument = docReader.getAnnotation(doc);
            if (timing != null) timing.report("Load coremap for " + docId);
            if (!coremapDocument.isDefined()) { return null; }

            // Pass to do any re-annotation
            if (reannotatePipeline != null) {
              if (timing != null) timing.start();
              reannotatePipeline.annotate(coremapDocument.get());
              if (timing != null) timing.report("Re-annotate " + docId);
            }

            // -- Set auxiliary annotations
            // These are set both on the document, and on each of the sentences in the document
            CoreMap[] coremapsToAnnotate = coremapDocument.get().get(SentencesAnnotation.class).toArray(new CoreMap[coremapDocument.get().get(SentencesAnnotation.class).size() + 1]);
            coremapsToAnnotate[coremapsToAnnotate.length - 1] = coremapDocument.get();
            // (get fields))
            Maybe<String> datetime = Maybe.Nothing();
            String dateTimeOrNull = doc.get(KBPField.DATETIME.fieldName());
            if (dateTimeOrNull != null) { datetime = Maybe.Just(dateTimeOrNull.trim()); }
            // (annotate)
            for (CoreMap toAnnotate : coremapsToAnnotate) {
              // Set source
              for (File directory : indexDirectory) {
                toAnnotate.set(KBPAnnotations.SourceIndexAnnotation.class, directory.getPath());
              }
              // Set Doc ID
              for (String id : docId) {
                toAnnotate.set(DocIDAnnotation.class, id);
              }
              // Set DateTime
              for (String date : datetime) {
                toAnnotate.set(KBPAnnotations.DatetimeAnnotation.class, date);
              }
              // Set Lucene DocID
              toAnnotate.set(KBPAnnotations.SourceIndexDocIDAnnotation.class, luceneId);
            }
            // Return
            return Pair.makePair(coremapDocument.get(), docPair.second);
            // --- CODE ENDS HERE ---

          } catch (IOException e) {
            logger.err(e);
            return null;
          } catch (ClassNotFoundException e) {
            logger.err(e);
            return null;
          }
        }
      };

  private static Function<Pair<Annotation, Double>, Iterator<Pair<CoreMap, Double>>>
  fetchSentences(final KBPEntity entity, final Maybe<KBPEntity> slotValue,
                 final Maybe<Integer> maxDocuments,
                 final AtomicInteger docsSeen) {
    final PostIRAnnotator postIRAnnotator;
    if (slotValue.isDefined()) {
      postIRAnnotator = new PostIRAnnotator(
          entity instanceof KBPOfficialEntity ? (KBPOfficialEntity) entity : KBPNew.from(entity).KBPOfficialEntity(),
          slotValue.get(), true);
    } else {
      postIRAnnotator = new PostIRAnnotator(
          entity instanceof KBPOfficialEntity ? (KBPOfficialEntity) entity : KBPNew.from(entity).KBPOfficialEntity(),
          true);
    }
    return new Function<Pair<Annotation, Double>, Iterator<Pair<CoreMap, Double>>>() {
      @Override
      public Iterator<Pair<CoreMap, Double>> apply(final Pair<Annotation, Double> docPair) {
        try {
          // --- CODE BEGINS HERE ---
          Annotation document = docPair.first;
          final List<CoreMap> sentences = document.get(SentencesAnnotation.class);
          Set<Integer> goodSentences = new HashSet<>();

          if (Props.INDEX_POSTIRANNOTATOR_DO) {
            // Annotate document
            postIRAnnotator.annotate(document);
            // Select relevant sentences
            for (int i = 0; i < sentences.size(); i++) {
              CoreMap sentence = sentences.get(i);
              if (!sentence.containsKey(KBPAnnotations.AllAntecedentsAnnotation.class)) { continue; }
              Set<String> antecedents = sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class);
              if (antecedents != null && antecedents.contains(entity.name) &&  // must contain entity name
                  (!slotValue.isDefined() || antecedents.contains(slotValue.get().name) || sentence.get(TextAnnotation.class).contains(slotValue.get().name)) &&  // must contain slot value (if defined)
                  (Props.INDEX_COREF_DO || (sentence.containsKey(KBPAnnotations.IsCoreferentAnnotation.class) &&!sentence.get(KBPAnnotations.IsCoreferentAnnotation.class)))) {  // maybe filter coreferent sentences
                goodSentences.add(i);
              }
            }
          }

          // Map good sentences back to their CoreMaps
          logger.debug("[" + docsSeen.incrementAndGet() + " / " + maxDocuments.getOrElse(-1) + "] " + goodSentences.size() + " sentences found: " + docPair.first.get(DocIDAnnotation.class));
          return CollectionUtils.mapIgnoreNull(goodSentences.iterator(), in -> {
            if (in == null) { logger.warn("null sentence index"); return null; }
            if (in >= sentences.size() || in < 0) { logger.warn("sentence index is out of bounds"); return null; }
            logger.debug("found sentence: " + CoreMapUtils.sentenceToMinimalString(sentences.get(in)));
            return Pair.makePair(sentences.get(in), docPair.second);
          });
          // --- CODE ENDS HERE ---

        } catch (RuntimeException e) {
          e.printStackTrace();
          logger.err(e);
          return null;
        }
      }
    };
  }

  /** A list of fields that need to be loaded to get a document from a lucene id */
  private static final Set<String> fieldsLuceneIdToDoc = new HashSet<>(
          Arrays.asList(new String[]{ "date", KBPField.DOCID.fieldName(), KBPField.DATETIME.fieldName(),
                  KBPField.COREMAP.fieldName(), KBPField.COREMAP_VERSION.fieldName(), KBPField.COREMAP_FILE.fieldName() }));

  protected static final Pattern REGEX_URL = Pattern.compile(
    "((([A-Za-z]{3,9}:(?://)?)(?:[-;:&=\\+\\$,\\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\\+\\$,\\w]+@)[A-Za-z0-9.-]+)((?:/[\\+~%/.\\w_-]*)?\\??(?:[-\\+=&;%@.\\w_]*)#?(?:[.!/\\w]*))?)");
}
