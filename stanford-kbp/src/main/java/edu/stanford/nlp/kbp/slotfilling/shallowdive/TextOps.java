package edu.stanford.nlp.kbp.slotfilling.shallowdive;

import au.com.bytecode.opencsv.CSVReader;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.entitylinking.EntityLinkingFeaturizer;
import edu.stanford.nlp.kbp.slotfilling.classify.OpenIERelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.ReverbRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.evaluate.HeuristicSlotfillPostProcessors;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.process.Feature;
import edu.stanford.nlp.kbp.slotfilling.process.Featurizable;
import edu.stanford.nlp.kbp.slotfilling.process.SlotMentionAnnotator;
import edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer;
import edu.stanford.nlp.kbp.slotfilling.train.KryoDatumCache;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.time.SUTimeSimpleParser;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.time.Timex;
import edu.stanford.nlp.util.*;

import java.io.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

import edu.stanford.nlp.util.logging.Redwood;
import gnu.trove.TObjectDoubleHashMap;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * TODO(gabor) JavaDoc
 *
 * @author Gabor Angeli
 */
public class TextOps {

  private static final Redwood.RedwoodChannels logger = Redwood.channels("TextOps");
  private static final Set<String> FIELDS_TO_LOAD = new HashSet<>(Arrays.<String>asList("word", "article", "score"));
  private static final Pattern NUMERIC = Pattern.compile("\\s*[0-9\\+:\\-\\.\\\\/\\* ]+\\s*");

  public final KBPIR ir;
  public final IndexSearcher searcher;
  private final Map<String, String> wikidictCached;

  /** A set of common first names */
  private final Set<String> firstNames;
  /** The annotated sentences from active learning */
  private final Map<String, String> annotationForSentence = new LinkedHashMap<>();

  public TextOps(KBPIR ir) {
    this.ir = ir;
    // Initialize WikiDict
    try {
      DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Props.INDEX_WIKIDICT));
      this.searcher = new IndexSearcher(reader);
      if (Props.SHALLOWDIVE_CACHEWIKIDICT) {
        forceTrack("Caching WikiDict");
        final Bits liveDocs = MultiFields.getLiveDocs(reader);
        final TObjectDoubleHashMap<String> linkScore = new TObjectDoubleHashMap<>();
        final HashMap<String, String> links = new HashMap<>();
        CollectionUtils.seqIter(reader.maxDoc()).forEachRemaining((luceneDocId) -> {
          if (liveDocs == null || liveDocs.get(luceneDocId)) {
            try {
              Document topDoc = searcher.doc(luceneDocId, FIELDS_TO_LOAD);
              double score = Double.parseDouble(topDoc.get("score"));
              String source = topDoc.get("word").toLowerCase();
              if (!linkScore.containsKey(source) || linkScore.get(source) < score) {
                linkScore.put(source, score);
                String target = topDoc.get("article").replaceAll("_", " ").replaceAll("\\(.*\\)", "").trim();
                links.put(source, target);
              }
            } catch (IOException e) {
              throw new RuntimeException(e);
            }
          }
        });
        this.wikidictCached = Collections.unmodifiableMap(links);
        endTrack("Caching WikiDict");
      } else {
        this.wikidictCached = null;

      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    // Read names
    Set<String> firstNames = new HashSet<>();
    for (Map.Entry<String,String> entry : EntityLinkingFeaturizer.readNicknames(Props.ENTITYLINKING_MALENAMES.getPath()).entrySet()) {
      firstNames.add(entry.getKey());
      firstNames.add(entry.getValue());
    }
    for (Map.Entry<String,String> entry : EntityLinkingFeaturizer.readNicknames(Props.ENTITYLINKING_FEMALENAMES.getPath()).entrySet()) {
      firstNames.add(entry.getKey());
      firstNames.add(entry.getValue());
    }
    this.firstNames = firstNames;

    // Read annotated sentences // TODO(gabor) copied code from KBPTrainer; should factor out at some point.
    // (read keys, if applicable)
    Maybe<? extends Set<String>> validKeys = Maybe.Nothing();
    if (Props.TRAIN_ANNOTATED_SENTENCES_KEYS != null && !Props.TRAIN_ANNOTATED_SENTENCES_KEYS.getPath().trim().equals("") && !Props.TRAIN_ANNOTATED_SENTENCES_KEYS.getPath().equals("/dev/null")) {
      try {
        validKeys = Maybe.Just(new HashSet<String>(Arrays.asList(IOUtils.slurpFile(Props.TRAIN_ANNOTATED_SENTENCES_KEYS).split("\n"))));
      } catch (IOException e) {
        logger.err("could not read annotated sentence keys file at " + Props.TRAIN_ANNOTATED_SENTENCES_KEYS + "; disallowing any annotated sentences");
        validKeys = Maybe.Just(new HashSet<String>());
      }
    }
    // (read data)
    for (File file : Props.TRAIN_ANNOTATED_SENTENCES_DATA) {
      try {
        CSVReader reader = new CSVReader(new FileReader(file));
        @SuppressWarnings("UnusedAssignment") // ignore the header
            String[] nextLine = reader.readNext();
        while ( (nextLine = reader.readNext()) != null ) {
          String key = nextLine[0];
          if (!validKeys.isDefined() || validKeys.get().contains(key)) {
            annotationForSentence.put(key, "no_relation".equals(nextLine[1]) ? RelationMention.UNRELATED : RelationType.fromString(nextLine[1]).orCrash("Unknown annotated relation type: " + nextLine[1]).canonicalName);
          }
        }
      } catch (IOException e) {
        logger.err("Could not read annotation file: " + file.getPath());
      }
    }
    logger.log("read " + annotationForSentence.size() + " labelled sentence annotations");
  }

  public Collection<Mention> mentions(CoreMap sentence) {
    // Setup
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

    // Augment NER tags for coreferent mentions
    for (CoreLabel token : tokens) {

      // Try to get the NER for the antecedent
      // 3 conditions: It should have an antecedent, the antecedent should be a proper noun, and the word should be a verifiable entity.
      // This last condition means that it is either:
      //   - A personal pronoun ( -> PERSON )
      //   - A location, as per world knowledge
      String antecedent = token.get(CoreAnnotations.AntecedentAnnotation.class);
      if (antecedent != null && antecedent.length() == 0) { antecedent = null; }
      if (token.ner().equals(Props.NER_BLANK_STRING) && token.tag().equals("PRP") && antecedent != null &&
          Character.isUpperCase(antecedent.charAt(0))) {
        if (SlotMentionAnnotator.personPronouns.contains(token.word().toLowerCase())) {
          token.setNER(NERTag.PERSON.name);
        } else if (Utils.geography().isValidCity(antecedent)) {
          token.setNER(NERTag.CITY.name);
        } else if (Utils.geography().isValidRegion(antecedent)) {
          token.setNER(NERTag.STATE_OR_PROVINCE.name);
        } else if (Utils.geography().isValidCountry(antecedent)) {
          token.setNER(NERTag.COUNTRY.name);
        }
      }
    }

    // Find slot mentions by iterating over tokens
    List<Mention> slots = new ArrayList<>();
    for (int start = 0; start < tokens.size(); ++start) {
      CoreLabel token = tokens.get(start);
      String ner = token.ner();
      String pos = token.tag();
      String antecedent = token.get(CoreAnnotations.AntecedentAnnotation.class);

      // valid candidates must be NEs, not the query entity, starting on a reasonable POS
      if (ner == null || "".equals(ner) || ner.equals("O") || pos.equals("IN") || pos.equals("DT") || pos.equals("RB") || pos.equals("EX") ||
          pos.equals("POS")) {
        continue;
      }

      //  tokens.get(start).word());
      int end = start + 1;
      while (end < tokens.size()) {
        CoreLabel crt = tokens.get(end);
        if (antecedent == null) { antecedent = crt.get(CoreAnnotations.AntecedentAnnotation.class); }
        if (antecedent == null) {
          Timex timex = crt.get(TimeAnnotations.TimexAnnotation.class);
          if (timex != null) { antecedent = timex.value(); }
        }
        if (crt.ner() == null || !crt.ner().equals(ner)) {
          break;
        }
        end++;
      }

      // fix up last POS, if invalid
      while (end > start + 1 && end > 0 &&  (tokens.get(end - 1).tag().equals("IN") || tokens.get(end - 1).tag().equals("DT") ||
          tokens.get(end - 1).tag().equals("RB") || tokens.get(end - 1).tag().equals("EX") || tokens.get(end - 1).tag().equals("POS"))) {
        end -= 1;
      }

      // if not valid, move on
      for (NERTag nerTag : NERTag.fromString(ner)) {
        Span span = new Span(start, end);
        assert !ner.trim().equalsIgnoreCase("") && !ner.equals(Props.NER_BLANK_STRING);
        // Resolve coref
        String name = antecedent;
        if (name == null) {
          name = CoreMapUtils.sentenceSpanString(sentence, span);
        }
        // Resolve entity linking
        if (nerTag == NERTag.PERSON || nerTag == NERTag.ORGANIZATION) {
          name = entityLink(name);
        }
        // Create mention
        Mention em  = new Mention(sentence, span, KBPNew.entName(name).entType(ner).KBPEntity());
        if (Props.KBP_VERBOSE) { logger.debug("found mention: " + em); }

        // Error check
        // (we don't actually care about long numbers)
        if (em.entity.type == NERTag.NUMBER && NUMERIC.matcher(em.entity.name).matches() && em.entity.name.length() > 3) { continue; }
        // ('one' is a common pronoun too)
        if (em.entity.type == NERTag.NUMBER && em.entity.name.equalsIgnoreCase("one")) { continue; }
        // (force geographic types for locations)
        if (!em.entity.type.isGeographic() &&
            (Utils.geography().isValidCountry(em.entity.name) || Utils.geography().isValidRegion(em.entity.name))) { continue; }
        // (disallow single word names)
        if (em.entity.type == NERTag.PERSON && !em.entity.name.contains(" ")) { continue; }
        // (dates must be absolute dates)
        if (em.entity.type == NERTag.DATE && !(
            HeuristicSlotfillPostProcessors.ConformToGuidelinesFilter.YEAR.matcher(em.entity.name).matches() ||
                HeuristicSlotfillPostProcessors.ConformToGuidelinesFilter.YEAR_MONTH.matcher(em.entity.name).matches() ||
                HeuristicSlotfillPostProcessors.ConformToGuidelinesFilter.YEAR_ONLY.matcher(em.entity.name).matches()
            )) { continue; }

        // Add
        slots.add(em);
      }

      start = end - 1;
    }
    return slots;
  }

  public Collection<Pair<Mention, Mention>> relationMentions(CoreMap sentence) {
    Collection<Mention> mentions = mentions(sentence);
    Collection<Pair<Mention, Mention>> relationMentions = new ArrayList<>();
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

    // Scan through entities
    for (Mention entity : mentions) {
      if (entity.entity.type.isEntityType()) {
        // Error check
        // (disallow entities which are too long or too short)
        if (entity.entity.name.length() > 50 || entity.entity.name.length() < 2) { continue; }
        // (disallow empty entities)
        if (entity.entity.name.trim().equals("")) { continue; }
        // (disallow entities that are just numbers)
        if (NUMERIC.matcher(entity.entity.name).matches()) { continue; }
        // (disallow entities that are times)
        if (tokens.get(entity.spanInSentence.start()).get(TimeAnnotations.TimexAnnotation.class) != null) { continue; }
        // (disallow single-name people -- remember, this is post-linking)
        if (entity.entity.type == NERTag.PERSON && !entity.entity.name.contains(" ")) { continue; }

        // Scan through slots
        for (Mention slot : mentions) {
          // Error check
          // (disallow long or short slot values)
          if (slot.entity.name.length() > 50 || slot.entity.name.length() < 2) { continue; }
          // (disallow empty slot values)
          if (slot.entity.name.trim().equals("")) { continue; }
          // (disallow self-references)
          if (slot.entity.name.equals(entity.entity.name) && slot.entity.type == entity.entity.type) { continue; }
          // (disallow impossible pairs)
          if (!RelationType.plausiblyHasRelation(entity.entity.type, slot.entity.type)) { continue; }

          // Add relation mention
          if (slot != entity && Utils.closeEnough(slot.spanInSentence, Collections.singleton(entity.spanInSentence))) {
            relationMentions.add(Pair.makePair(entity, slot));
          }

        }
      }
    }
    return relationMentions;
  }

  public String entityLink(String personOrOrganization) {
    try {
      // Case: common first name
      if (!personOrOrganization.contains(" ")) {
        if (firstNames.contains(personOrOrganization.toLowerCase())) {
          return personOrOrganization;
        }
      }
      if (this.wikidictCached != null) {
        String canonicalName = this.wikidictCached.get(personOrOrganization.toLowerCase());
        if (canonicalName != null) {
          return canonicalName;
        } else {
          return personOrOrganization;
        }
      } else {
        // Issue query
        Query query = new TermQuery(new Term("word", personOrOrganization));
        TopFieldDocs results = this.searcher.search(query, 2, Sort.RELEVANCE);
        // Case: no results
        if (results.totalHits == 0) {
          return personOrOrganization;
        }
        // Case: only one hit
        ScoreDoc topHit = results.scoreDocs[0];
        Document topDoc = searcher.doc(topHit.doc, FIELDS_TO_LOAD);
        if (Double.parseDouble(topDoc.get("score")) < 0.4) {
          return personOrOrganization;
        }
        if (results.totalHits == 1) {
          return topDoc.get("article").replaceAll("_", " ").replaceAll("\\(.*\\)", "").trim();
        }
        // Case: multiple results
        ScoreDoc secondHit = results.scoreDocs[1];
        Document secondDoc = searcher.doc(secondHit.doc, FIELDS_TO_LOAD);
        if (Double.parseDouble(topDoc.get("score")) > Double.parseDouble(secondDoc.get("score")) * 1.5) {
          return topDoc.get("article").replaceAll("_", " ").replaceAll("\\(.*\\)", "").trim();
        } else {
          return personOrOrganization;
        }
      }
    } catch (IOException e) {
      logger.err(e);
      return personOrOrganization;
    }
  }


  /**
   * Entity link a {@link edu.stanford.nlp.kbp.common.KBTriple}, both the entity and slot value.
   *
   * @see TextOps#entityLink(String)
   *
   * @param in The triple being linked.
   * @return A copy of this triple, appropriately linked.
   */
  public KBTriple linkTriple(KBTriple in) {
    //noinspection LoopStatementThatDoesntLoop
    for (NERTag slotType : in.slotType.orElse(Utils.inferFillType(in.kbpRelation()))) {
      KBPNew.KBTripleBuilder builder = KBPNew.from(in);
      if (slotType == NERTag.DATE) {
        try {
          String timexValue = SUTimeSimpleParser.parse(in.slotValue).getTimexValue();
          if (timexValue != null) {
            builder = builder.slotValue(timexValue);
          } else {
            builder = builder.slotValue(entityLink(in.slotValue));
          }
        } catch (SUTimeSimpleParser.SUTimeParsingError ignored) { }
      } else {
        builder = builder.slotValue(entityLink(in.slotValue));
      }
      return builder
          .entName(entityLink(in.entityName))
          .slotType(in.slotType.getOrElse(Utils.inferFillType(in.kbpRelation()).orCrash())).KBTriple();
    }
    throw new IllegalStateException("Could not determine slot type for training triple: " + in);
  }

  /**
   * Link the entire input knowledge base.
   * This is necessary to query the database, otherwise there's a mismatch between how the database was stored
   * and how it's being queried.
   *
   * @see TextOps#linkTriple(edu.stanford.nlp.kbp.common.KBTriple)
   * @see edu.stanford.nlp.kbp.slotfilling.shallowdive.TextOps#entityLink(String)
   *
   * @param in The knowledge base being linked (see, e.g., {@link edu.stanford.nlp.kbp.slotfilling.ir.KBPIR#trainingTriples()}).
   * @return A linked version of the knowledge base, to be used as input to, e.g., {@link edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer#makeDataset(java.util.Iterator, edu.stanford.nlp.kbp.common.Maybe)}.
   */
  public Map<KBPair, Set<String>> linkKB(Collection<KBTriple> in) {
    forceTrack("Linking KB");
    Map<KBPair, Set<String>> linkedKB = new HashMap<>();
    for (KBTriple triple : in) {
      KBTriple linkedTriple = linkTriple(triple);
      KBPair key = KBPNew.from(linkedTriple).entId(Maybe.<String>Nothing()).KBPair();
      if (!linkedKB.containsKey(key)) {
        linkedKB.put(key, new HashSet<>());
      }
      linkedKB.get(key).add(linkedTriple.relationName);
    }
    endTrack("Linking KB");
    return linkedKB;
  }

  public static interface SentenceCallback<E> {
    public void apply(E metadata, Annotation document, CoreMap sentence, List<KBPSlotFill> openieExtractions);
  }

  public <E> void applyToEverySentence(final Factory<E> createData,
                                       final Function<E, Exception> destroyData,
                                       final SentenceCallback<E> fn) {
    applyToEverySentence(createData, destroyData, fn, Integer.MAX_VALUE);
  }

  public <E> void applyToEverySentence(final Factory<E> createData,
                                       final Function<E, Exception> destroyData,
                                       final SentenceCallback<E> fn, int maxDocuments) {
    final AtomicInteger docCount = new AtomicInteger(0);
    final Map<Long, OpenIERelationExtractor> openie = new HashMap<>();
    ir.slurpDocuments(maxDocuments).filter(in ->
          in != null && in.get(CoreAnnotations.TokensAnnotation.class) != null &&
            in.get(CoreAnnotations.TokensAnnotation.class).size() < 5000).forEach( document -> {
            try {
            E statement = createData.create();
            int id = docCount.getAndIncrement();

            // Process OpenIE
            List<KBPSlotFill> openieExtractions = Collections.EMPTY_LIST;
            if (Props.SHALLOWDIVE_FEATURIZE_OPENIE) {
              new SlotMentionAnnotator().annotate(document);  // TODO(gabor) needed for the OpenIE extractor, but otherwise a bit nasty to have here
              try {
                OpenIERelationExtractor openieExtractor;
                synchronized (openie) {
                  if (!openie.containsKey(Thread.currentThread().getId())) {
                    openie.put(Thread.currentThread().getId(), new ReverbRelationExtractor());
                  }
                  openieExtractor = openie.get(Thread.currentThread().getId());
                }
                openieExtractions = openieExtractor.extractRelations(document);
              } catch (Exception ignored) {
              }
            }

            // Process sentences
            try {
              for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
                // Discard awful sentences
                if (sentence.get(CoreAnnotations.TokensAnnotation.class).size() > 50) {
                  logger.warn("[" + id + "] ignoring sentence of length > 50");
                  continue;
                }
                SemanticGraph dependencyGraph = sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
                if (dependencyGraph == null) {
                  logger.warn("[" + id + "] malformed semantic graph for sentence");
                  continue;
                }
                // -- APPLY FUNCTION --
                try {
                  fn.apply(statement, document, sentence, openieExtractions);
                } catch (Exception e) {
                  logger.log(e);
                }
                // --                --
              }
            } catch (Exception e) {
              logger.err(e);
            }

            // Flush
            synchronized (statement) {
              try {
                Exception t = destroyData.apply(statement);
                if (t != null) {
                  throw t;
                }
              } catch (SQLException e) {
                logger.err(e);
                logger.err(e.getNextException());
              } catch (Exception e) {
                logger.err(e);
              }
            }
          } catch (Throwable t) {
            logger.err(t);
          }
        });
  }

  public void applyToEverySentence(SentenceCallback<Class<Void>> callback) {
    applyToEverySentence(
        () -> Void.class,
        in -> null,
        callback
    );
  }


  private void beforeFeaturizeToTable(final String tableName,
                                      Pointer<Factory<PreparedStatement>> insertStatementFactory,
                                      Pointer<Factory<PreparedStatement>> trueRelationInsertFactory,
                                      Pointer<Factory<PreparedStatement>> falseRelationInsertFactory
                                      ) {
    forceTrack("Setting up table: " + tableName);
    PostgresUtils.dropTable(tableName, true);
    PostgresUtils.dropTable(tableName + "_relations", true);
    PostgresUtils.withConnection(tableName, psql -> {

      // Main datum table
      psql.createStatement().execute("CREATE TABLE " + tableName + " (" +
          "did BIGINT PRIMARY KEY, " +
          "entity_name TEXT, " +
          "entity_type TEXT, " +
          "slot_value TEXT, " +
          "slot_value_type TEXT, " +
          "document_id TEXT, " +
          "sentence_index INT, " +
          "entity_span_start INT, " +
          "entity_span_length INT, " +
          "slot_value_span_start INT, " +
          "slot_value_span_end INT, " +
          "datum BYTEA );");
      insertStatementFactory.set(() -> {
        try {
          return psql.prepareStatement("INSERT INTO " + tableName + "" +
              "(did, entity_name, entity_type, slot_value, slot_value_type, document_id, sentence_index, " +
              " entity_span_start, entity_span_length, slot_value_span_start, slot_value_span_end, datum) " +
              "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);");
        } catch (SQLException e) {
          throw new RuntimeException(e);
        }
      });

      // Relations table
      psql.createStatement().execute("CREATE TABLE " + tableName + "_relations" + " (" +
          "did BIGINT, " +
          "truth BOOLEAN, " +
          "relation_name TEXT);");

      // True relations insert
      trueRelationInsertFactory.set(() -> {
        try {
          return psql.prepareStatement("INSERT INTO " + tableName + "_relations " +
              "(did, truth, relation_name) " +
              "VALUES(?, true, ?);");
        } catch (SQLException e) {
          throw new RuntimeException(e);
        }
      });

      // False relations insert
      falseRelationInsertFactory.set(() -> {
        try {
          return psql.prepareStatement("INSERT INTO " + tableName + "_relations " +
              "(did, truth, relation_name) " +
              "VALUES(?, false, ?);");
        } catch (SQLException e) {
          throw new RuntimeException(e);
        }
      });
    });
    endTrack("Setting up table: " + tableName);
  }

  public void afterFeaturizeToTable(final String tableName) {
    forceTrack("Creating indices");
    PostgresUtils.withConnection(tableName, psql -> {
      psql.createStatement().execute("CREATE INDEX " + tableName + "_kbpair ON "
          + tableName + " USING BTREE (entity_name, entity_type, slot_value, slot_value_type);");
      psql.createStatement().execute("CREATE INDEX " + tableName + "_relations_did ON "
          + tableName + "_relations USING HASH (did);");
      psql.createStatement().execute("CREATE INDEX " + tableName + "_relations_relation ON "
          + tableName + "_relations USING BTREE (relation_name);");
      psql.createStatement().execute("CREATE INDEX " + tableName + "_relations_truth ON "
          + tableName + "_relations USING BTREE (truth);");
    });
    endTrack("Creating indices");
  }



  public void featurizeToTable(final String tableName) {
    // Pre-heat table (in single threaded mode)
    final Pointer<Factory<PreparedStatement>> insertStatementFactory = new Pointer<>();
    final Pointer<Factory<PreparedStatement>> trueRelationInsertFactory = new Pointer<>();
    final Pointer<Factory<PreparedStatement>> falseRelationInsertFactory = new Pointer<>();
    beforeFeaturizeToTable(tableName, insertStatementFactory, trueRelationInsertFactory, falseRelationInsertFactory);

    final Map<KBPair, Set<String>> kb = linkKB(ir.trainingTriples());
    final AtomicLong id = new AtomicLong(0);

    Map<Long, Triple<PreparedStatement, PreparedStatement, PreparedStatement>> statements = new ConcurrentHashMap<>();
    Map<Long, AtomicInteger> queueSize = new ConcurrentHashMap<>();
    applyToEverySentence(

        // -- Create New Statements --
        () -> {
          Triple<PreparedStatement, PreparedStatement, PreparedStatement> triple = statements.get(Thread.currentThread().getId());
          if (triple == null) {
            triple = Triple.makeTriple(
                insertStatementFactory.dereference().orCrash().create(),
                trueRelationInsertFactory.dereference().orCrash().create(),
                falseRelationInsertFactory.dereference().orCrash().create());
            statements.put(Thread.currentThread().getId(), triple);
            queueSize.put(Thread.currentThread().getId(), new AtomicInteger());
          }
          return triple;
        },

        // -- Flush Statements --
        in -> {
          try {
            if (queueSize.get(Thread.currentThread().getId()).incrementAndGet() % 10000 == 0) {
              in.first.executeBatch();
              in.second.executeBatch();
              in.third.executeBatch();
            }
            return null;
          } catch (Exception e) {
            return e;
          }
        },

        // -- Process Code --
        (inserts, document, sentence, openieExtractions) -> {
          PreparedStatement insert = inserts.first;
          PreparedStatement trueRelation = inserts.second;
          PreparedStatement falseRelation = inserts.third;
          Collection<Pair<Mention, Mention>> relationMentionList = relationMentions(sentence);
//        if (relationMentionList.size() > 0) {
//          logger.log("" + relationMentionList.size() + " relation mentions in sentence " + sentence.get(CoreAnnotations.SentenceIndexAnnotation.class));
//        }

          // Process relation mentions
          for (Pair<Mention, Mention> relationMention : relationMentionList) {
            // Find matching OpenIE extractions
            List<KBPSlotFill> matchingExtractions = new ArrayList<>();
            for (KBPSlotFill candidate : openieExtractions) {
              for (KBPRelationProvenance provenance : candidate.provenance) {
                if (provenance.sentenceIndex.getOrElse(-1).equals(sentence.get(CoreAnnotations.SentenceIndexAnnotation.class))) {
                  for (Span entitySpan : provenance.entityMentionInSentence) {
                    for (Span slotSpan : provenance.slotValueMentionInSentence) {
                      if ((Span.overlaps(relationMention.first.spanInSentence, entitySpan) && Span.overlaps(relationMention.second.spanInSentence, slotSpan)) ||
                          (Span.overlaps(relationMention.second.spanInSentence, entitySpan) && Span.overlaps(relationMention.first.spanInSentence, slotSpan))) {
                        matchingExtractions.add(candidate);
                      }
                    }
                  }
                }
              }
            }
            // Featurize
            Span subj = relationMention.first.spanInSentence;
            Span obj = relationMention.second.spanInSentence;
            List<CoreLabel> tokens = relationMention.first.sentence.get(CoreAnnotations.TokensAnnotation.class);
            SemanticGraph dependencies = relationMention.first.sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
            Featurizable factory = new Featurizable(subj, obj, tokens, dependencies, matchingExtractions);
            Counter<String> features = new ClassicCounter<>();
            for (Feature feat : Feature.values()) {
              feat.provider.apply(factory, features);
            }

            // Create singleton sentence group
            // (key)
            KBPair key = KBPNew.from(relationMention.first.entity).slotValue(relationMention.second.entity).KBPair();
            // (datum)
            Datum<String, String> datum = new BasicDatum<>(Counters.toSortedList(features));
            // (provenance)
            KBPRelationProvenance provenance = new KBPRelationProvenance(
                document.get(CoreAnnotations.DocIDAnnotation.class),
                document.get(KBPAnnotations.SourceIndexAnnotation.class),
                sentence.get(CoreAnnotations.SentenceIndexAnnotation.class),
                relationMention.first.spanInSentence,
                relationMention.second.spanInSentence,
                sentence);
            // (sentence gloss key)
            String hexKey = CoreMapUtils.getSentenceGlossKey(sentence.get(CoreAnnotations.TokensAnnotation.class), relationMention.first.spanInSentence, relationMention.second.spanInSentence);
            // (create)
            final SentenceGroup group = new SentenceGroup(key, datum, provenance, hexKey);

            try {
              // Insert Datum
              long did = id.incrementAndGet();
              insert.setLong(1, did);
              insert.setString(2, key.entityName);
              insert.setString(3, key.entityType.name);
              insert.setString(4, key.slotValue);
              insert.setString(5, key.slotType.orCrash().name);
              insert.setString(6, provenance.docId);
              insert.setInt(7, provenance.sentenceIndex.orCrash());
              insert.setInt(8, provenance.entityMentionInSentence.orCrash().start());
              insert.setInt(9, provenance.entityMentionInSentence.orCrash().end() - provenance.entityMentionInSentence.orCrash().start());
              insert.setInt(10, provenance.slotValueMentionInSentence.orCrash().start());
              insert.setInt(11, provenance.slotValueMentionInSentence.orCrash().end() - provenance.slotValueMentionInSentence.orCrash().start());
              // (save datum)
              ByteArrayOutputStream out = new ByteArrayOutputStream();
              KryoDatumCache.save(group, out);
              byte[] data = out.toByteArray();
              ByteArrayInputStream is = new ByteArrayInputStream(out.toByteArray());
              insert.setBinaryStream(12, is, data.length);
              is.close();
              out.close();
              insert.addBatch();

              // Insert Relations
              Pair<Set<String>, Set<String>> relations = KBPTrainer.computePositiveAndNegativeRelations(
                  group,
                  ir,
                  annotationForSentence::get,
                  kb::get);
              for (String positiveRelation : relations.first) {
                trueRelation.setLong(1, did);
                trueRelation.setString(2, positiveRelation);
                trueRelation.addBatch();
              }
              for (String negativeRelation : relations.second) {
                falseRelation.setLong(1, did);
                falseRelation.setString(2, negativeRelation);
                falseRelation.addBatch();
              }
            } catch (SQLException | IOException e) {
              logger.log(e);
            }
          }
        }
    );

    // Clean up
    for (Triple<PreparedStatement, PreparedStatement, PreparedStatement> triple : statements.values()) {
      try {
        triple.first.executeBatch();
        triple.second.executeBatch();
        triple.third.executeBatch();
      } catch (SQLException e) {
        logger.err(e);
      }
    }

    afterFeaturizeToTable(tableName);
  }
}
