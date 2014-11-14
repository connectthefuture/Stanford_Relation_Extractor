package edu.stanford.nlp.kbp.slotfilling.shallowdive;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet;
import edu.stanford.nlp.kbp.slotfilling.evaluate.SlotFiller;
import edu.stanford.nlp.kbp.slotfilling.evaluate.SlotfillPostProcessor;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.forceTrack;

/**
* A slot filler making use of the data populated using edu.stanford.nlp.kbp.slotfilling.shallowdive.ShallowDive.
*
* @author Gabor Angeli
*/
public class ShallowDiveSlotFiller implements SlotFiller {
  private static Redwood.RedwoodChannels logger = Redwood.channels("ShallowDiveSF");

  public final String kbTable;
  public final String datumTable;
  public final TextOps textOps;
  private final GoldResponseSet responseChecklist = new GoldResponseSet();

  private final Lazy<RelationClassifier> classifier;

  public ShallowDiveSlotFiller(String kbTable, String datumTable, KBPIR ir) {
    this.kbTable = kbTable;
    this.datumTable = datumTable;
    this.textOps = new TextOps(ir);
    this.classifier = new Lazy<RelationClassifier>() {
      @Override
      protected RelationClassifier compute() {
        return Props.TRAIN_MODEL.load(Props.KBP_MODEL_PATH, new Properties());
      }
    };
  }

  public Set<KBPSlotFill> classify(SentenceGroup datum) {
    Set<KBPSlotFill> predictions = new HashSet<>();
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> relations = classifier.get().classifyRelations(datum, Maybe.<CoreMap[]>Nothing());
    for (Pair<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : Counters.toSortedListWithCounts(relations)) {
      for (KBPRelationProvenance provenance : entry.first.second) {
        for (RelationType relation : RelationType.fromString(entry.first.first)) {
          double score = entry.second;
          // Check consistency
          KBPSlotFill fill = KBPNew
              .entName(datum.key.entityName)
              .entType(datum.key.entityType.name)
              .slotValue(datum.key.slotValue)
              .slotType(datum.key.slotType.getOrElse(Utils.inferFillType(relation).orNull()).name)
              .rel(relation.canonicalName)
              .score(score)
              .provenance(provenance).KBPSlotFill();
          predictions.add(fill);
        }
      }
    }
    return predictions;
  }

  /**
   * The ShallowDive consistency component.
   * @param queryEntity The query entity we are filtering slots for.
   * @param fromDB The set of raw slots, as returned from the database (or classified on the fly)
   * @return A list of consistent slot fills.
   */
  private List<KBPSlotFill> doConsistency(KBPOfficialEntity queryEntity, Set<KBPSlotFill> fromDB) {
    // Do consistency
    // (convert to sorted list)
    logger.log("" + fromDB.size() + " raw slots found");
    List<KBPSlotFill> sortedSlotFills = new ArrayList<>(fromDB);
    Collections.sort(sortedSlotFills);
    // (prune)
    if (sortedSlotFills.size() > 100) {
      sortedSlotFills = sortedSlotFills.subList(0, 100);
    }
    for (KBPSlotFill fill : sortedSlotFills) {
      this.responseChecklist.registerResponse(fill);
    }
    // (post-process)
    SlotfillPostProcessor processor = SlotfillPostProcessor.global(textOps.ir);
    List<KBPSlotFill> finalSlots = processor.postProcess(queryEntity, SlotfillPostProcessor.unary.postProcess(queryEntity, sortedSlotFills, responseChecklist), responseChecklist);
    logger.log("" + finalSlots.size() + " final slots proposed");
    // (log)
    logger.prettyLog(responseChecklist.loggableForEntity(queryEntity, Maybe.Just(textOps.ir)));
    return finalSlots;
  }

  /**
   * @param queryEntity The query entity to fill slots for.
   * @return The consistency-filtered slots for this query entity.
   */
  public List<KBPSlotFill> fillSlotsFromDB(final KBPOfficialEntity queryEntity) {
    forceTrack("Finding slots for " + queryEntity);
    final Set<KBPSlotFill> fromDB = new HashSet<>();
    PostgresUtils.withConnection(kbTable, psql -> {
      // Do query
      String query;
      try {
        query = "SELECT * FROM " + kbTable +
            " WHERE entity_name = '" + textOps.entityLink(queryEntity.name).replaceAll("'", "''") +
            "' AND entity_type = '" + queryEntity.type +
            "' AND source_index='" + Props.INDEX_OFFICIAL.getCanonicalFile().getAbsolutePath() + "';";
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      logger.log("query: " + query);
      ResultSet result = psql.createStatement().executeQuery(query);

      // Fill slots
      while (result.next()) {

        fromDB.add(KBPNew.from(queryEntity)
            .slotValue(result.getString("slot_value_name"))
            .slotType(result.getString("slot_value_type"))
            .rel(result.getString("relation"))
            .score(result.getDouble("score"))
            .provenance(new KBPRelationProvenance(
                    result.getString("doc_id"),
                    Props.INDEX_OFFICIAL.getPath(),  // always the official index
                    result.getInt("sentence_index"),
                    new Span(result.getInt("entity_token_begin"),
                        result.getInt("entity_token_begin") + result.getInt("entity_token_length")),
                    new Span(result.getInt("slot_value_token_begin"),
                        result.getInt("slot_value_token_begin") + result.getInt("slot_value_token_length"))
                )
            ).KBPSlotFill());
      }
    });

    // Do consistency
    // (convert to sorted list)
    logger.log("" + fromDB.size() + " raw slots found");
    List<KBPSlotFill> sortedSlotFills = new ArrayList<>(fromDB);
    Collections.sort(sortedSlotFills);
    // (prune)
    if (sortedSlotFills.size() > 100) {
      sortedSlotFills = sortedSlotFills.subList(0, 100);
    }
    for (KBPSlotFill fill : sortedSlotFills) {
      this.responseChecklist.registerResponse(fill);
    }
    // (post-process)
    SlotfillPostProcessor processor = SlotfillPostProcessor.global(textOps.ir);
    List<KBPSlotFill> finalSlots = processor.postProcess(queryEntity, SlotfillPostProcessor.unary.postProcess(queryEntity, sortedSlotFills, responseChecklist), responseChecklist);
    logger.log("" + finalSlots.size() + " final slots proposed");
    // (log)
    logger.prettyLog(responseChecklist.loggableForEntity(queryEntity, Maybe.Just(textOps.ir)));

    // Return
    endTrack("Finding slots for " + queryEntity);
    return finalSlots;
  }

  public List<KBPSlotFill> fillSlotsLazily(final KBPOfficialEntity queryEntity) {
    forceTrack("Finding slots for " + queryEntity + " (lazy)");
    final Set<KBPSlotFill> fromDB = new HashSet<>();
    PostgresUtils.withConnection(kbTable, psql -> {
      final PreparedStatement query = psql.prepareStatement("SELECT * FROM " + datumTable + " WHERE entity_name=? AND entity_type=? ORDER BY slot_value, slot_value_type DESC");
      // (issue query
      query.setString(1, textOps.entityLink(queryEntity.name));
      query.setString(2, queryEntity.type.name);
      ResultSet rs = query.executeQuery();
      // (start reading)
      if (!rs.next()) {
        return;
      }
      SentenceGroup currentGroup = DatumOps.mkDatum(rs);
      // (grok query)
      while (rs.next()) {
        SentenceGroup newGroup = DatumOps.mkDatum(rs);
        if (newGroup.key.equals(currentGroup.key)) {
          currentGroup.merge(newGroup);
        } else {
          currentGroup = currentGroup.removeDuplicateDatums();
          fromDB.addAll(classify(currentGroup));
          currentGroup = newGroup;
        }
      }
      fromDB.addAll(classify(currentGroup));
    });

    List<KBPSlotFill> finalSlots = doConsistency(queryEntity, fromDB);
    endTrack("Finding slots for " + queryEntity + " (lazy)");
    return finalSlots;
  }

  /** {@inheritDoc} */
  @Override
  public List<KBPSlotFill> fillSlots(final KBPOfficialEntity queryEntity) {
    if (Props.SHALLOWDIVE_EVALUATE_LAZY) {
      return fillSlotsLazily(queryEntity);
    } else {
      return fillSlotsFromDB(queryEntity);
    }
  }
}
