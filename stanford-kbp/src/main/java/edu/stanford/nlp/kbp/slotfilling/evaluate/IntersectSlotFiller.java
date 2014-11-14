package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.common.KBPOfficialEntity;
import edu.stanford.nlp.kbp.common.KBPSlotFill;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.forceTrack;

/**
 * A utility to compare two slot fillers -- canonically the inferential and simple slot fillers.
 *
 * @author Gabor Angeli
 */
public class IntersectSlotFiller implements SlotFiller {

  Redwood.RedwoodChannels logger = Redwood.channels("DiffFiller");

  private final SlotFiller a;
  private final SlotFiller b;

  public IntersectSlotFiller(SlotFiller a, SlotFiller b) {
    this.a = a;
    this.b = b;
  }

  @Override
  public List<KBPSlotFill> fillSlots(KBPOfficialEntity queryEntity) {
    Set<KBPSlotFill> slotsA = new HashSet<>(a.fillSlots(queryEntity));
    Set<KBPSlotFill> slotsB = new HashSet<>(b.fillSlots(queryEntity));

    Set<KBPSlotFill> intersection = new HashSet<>();
    for (KBPSlotFill fill : slotsA) {
      if (slotsB.contains(fill)) {
        intersection.add(fill);
      }
    }

    forceTrack("Diff Fills");
    forceTrack("Agree");
    for (KBPSlotFill fill : intersection) {
      logger.log(fill);
    }
    endTrack("Agree");
    forceTrack("A Only");
    for (KBPSlotFill fill : slotsA) {
      if (!intersection.contains(fill)) { logger.log(fill); }
    }
    endTrack("A Only");
    forceTrack("B Only");
    for (KBPSlotFill fill : slotsB) {
      if (!intersection.contains(fill)) { logger.log(fill); }
    }
    endTrack("B Only");
    endTrack("Diff Fills");

    return new ArrayList<>(intersection);
  }
}
