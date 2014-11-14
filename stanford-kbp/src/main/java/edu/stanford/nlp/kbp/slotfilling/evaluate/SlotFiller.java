package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.common.KBPOfficialEntity;
import edu.stanford.nlp.kbp.common.KBPSlotFill;

import java.util.List;

/**
 * @author Gabor Angeli
 */
public interface SlotFiller {

  public List<KBPSlotFill> fillSlots(KBPOfficialEntity queryEntity);

}
