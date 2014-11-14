package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A class storing useful functions for creating fake data for consistency processors
 *
 * @author Gabor Angeli
 */
public class PostProcessorsData {
  protected KBPEntity julie = KBPNew.entName("Julie").entType(NERTag.PERSON).KBPOfficialEntity();
  protected KBPEntity stanford = KBPNew.entName("Stanford").entType(NERTag.ORGANIZATION).KBPOfficialEntity();
  protected KBPEntity stanfordLong = KBPNew.entName("Leeland Stanford Junior University").entType(NERTag.ORGANIZATION).KBPOfficialEntity();
  protected KBPEntity kbpinc = KBPNew.entName("KBP Incorporated").entType(NERTag.ORGANIZATION).KBPOfficialEntity();
  protected KBPEntity hamilton = KBPNew.entName("Lewis Hamilton").entType(NERTag.PERSON).KBPOfficialEntity();

  protected KBPSlotFill fill(KBPEntity entity, RelationType rel, String value, double score) {
    return KBPNew.from(entity).entId("(null)").slotValue(value).rel(rel.canonicalName).score(score).KBPSlotFill();
  }

  protected KBPSlotFill fill(KBPEntity entity, RelationType rel, String value, NERTag tag, double score) {
    return KBPNew.from(entity).entId("(null)").slotValue(value).slotType(tag).rel(rel.canonicalName).score(score).KBPSlotFill();
  }

  protected KBPSlotFill fill(KBPEntity entity, RelationType rel, String value, NERTag valueType, double score,
                             int entityStart, int entityEnd, int slotStart, int slotEnd, CoreMap sentence) {
    return KBPNew.from(entity).slotValue(value).slotType(valueType).rel(rel)
        .provenance(new KBPRelationProvenance("(null)", "(null)", 0, new Span(entityStart, entityEnd), new Span(slotStart, slotEnd), sentence))
        .score(score).KBPSlotFill();
  }

  protected Map<KBPEntity, List<KBPSlotFill>> dummyData() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();

    data.put(julie, new ArrayList<KBPSlotFill>());
    // A valid relation
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", NERTag.COUNTRY, 0.8));
    // Incorrect relation type
    data.get(julie).add(fill(julie, RelationType.ORG_MEMBERS, "Chris Manning", NERTag.PERSON, 0.8));
    // Date without year
    data.get(julie).add(fill(julie, RelationType.PER_DATE_OF_BIRTH, "May 19", NERTag.DATE, 0.7));
    // TODO(gabor) test ignored slots
    // TODO(gabor) test already known slots
    // Duplicates
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", NERTag.COUNTRY, 0.7));
    data.get(julie).add(fill(julie, RelationType.PER_DATE_OF_BIRTH, "May 19", NERTag.DATE, 0.5));
    // List-valued duplicate
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "United States", NERTag.COUNTRY, 0.4)); // note score < Canada
    // Plausible co-occuring relations
    data.get(julie).add(fill(julie, RelationType.PER_EMPLOYEE_OF, "Canada", NERTag.COUNTRY, 0.4));
    data.get(julie).add(fill(julie, RelationType.PER_ALTERNATE_NAMES, "Canada", NERTag.COUNTRY, 0.3)); // note: score < COUNTRY_OF_BIRTH; < EMPLOYEE_OF
    // Mitigate location of death
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_DEATH, "Switzerland", NERTag.COUNTRY, 0.5));


    data.put(stanford, new ArrayList<KBPSlotFill>());
    // A mix of things to filter and keep
    data.get(stanford).add(fill(stanford, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", NERTag.COUNTRY, 0.6));
    data.get(stanford).add(fill(stanford, RelationType.ORG_MEMBERS, "Stanford CS", NERTag.ORGANIZATION, 0.8));

    return data;
  }

  protected Map<KBPEntity, List<KBPSlotFill>> urls() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();

    data.put(stanfordLong, new ArrayList<KBPSlotFill>());
    // Positive Examples
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://www.stanford.edu/", NERTag.URL, 1.0));
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://stanford.edu/", NERTag.URL, 1.0));
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://leelandstanfordjunioruniversity.edu/", NERTag.URL, 1.0));
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://leelandstanforduniversity.edu/", NERTag.URL, 1.0));
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://lsju.edu/", NERTag.URL, 1.0));
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://stanford.co.uk/", NERTag.URL, 1.0));
    // Negative Examples
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://www.foo.edu/", NERTag.URL, 1.0));
    data.get(stanfordLong).add(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://www.bar.edu/stanford", NERTag.URL, 1.0));

    return data;
  }

  protected Map<KBPEntity, List<KBPSlotFill>> reflexiveData() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();

    data.put(julie, new ArrayList<KBPSlotFill>());

    data.put(stanford, new ArrayList<KBPSlotFill>());
    // A mix of things to filter and keep
    data.get(stanford).add(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Stanford University", NERTag.ORGANIZATION, 0.6));
    data.get(stanford).add(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Julie", NERTag.PERSON, 0.6));

    return data;
  }

  protected Map<KBPEntity, List<KBPSlotFill>> orderingCornerCasesData() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();

    data.put(julie, new ArrayList<KBPSlotFill>());
    // (pairwise inconsistent)
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", NERTag.COUNTRY, 0.8));
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "France", NERTag.COUNTRY, 0.7));
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "21", NERTag.NUMBER, 0.6));
    // (inconsistent with country_of_birth)
    data.get(julie).add(fill(julie, RelationType.PER_AGE, "Canada", NERTag.COUNTRY, 0.7));
    data.get(julie).add(fill(julie, RelationType.PER_AGE, "21", NERTag.NUMBER, 0.65));

    return data;
  }

  protected Map<KBPEntity, List<KBPSlotFill>> approximateDuplicateData() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();

    data.put(julie, new ArrayList<KBPSlotFill>());  // note: most of these don't actually apply to the real Julie
    data.get(julie).add(fill(julie, RelationType.PER_SIBLINGS, "Adan Chavez", NERTag.PERSON, 1.0));
    data.get(julie).add(fill(julie, RelationType.PER_SIBLINGS, "Adan", NERTag.PERSON, 0.9));  // cut
    data.get(julie).add(fill(julie, RelationType.PER_TITLE, "singer/songwriter", NERTag.TITLE, 1.0));
    data.get(julie).add(fill(julie, RelationType.PER_TITLE, "Singer\\/songwriter", NERTag.TITLE, 0.9));  // cut
    data.get(julie).add(fill(julie, RelationType.PER_MEMBER_OF, "Socialist Party", NERTag.IDEOLOGY, 1.0));
    data.get(julie).add(fill(julie, RelationType.PER_MEMBER_OF, "United Socialist party", NERTag.IDEOLOGY, 0.9));  // cut
    data.get(julie).add(fill(julie, RelationType.PER_TITLE, "murder defendant", NERTag.CRIMINAL_CHARGE, 1.0));
    data.get(julie).add(fill(julie, RelationType.PER_TITLE, "defendant", NERTag.CRIMINAL_CHARGE, 0.9));  // cut
    data.get(julie).add(fill(julie, RelationType.PER_MEMBER_OF, "American Family Association", NERTag.ORGANIZATION, 1.0));
    data.get(julie).add(fill(julie, RelationType.PER_MEMBER_OF, "AFA", NERTag.ORGANIZATION, 0.9));  // cut


    data.put(stanford, new ArrayList<KBPSlotFill>());
    data.get(stanford).add(fill(stanford, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, "California", NERTag.STATE_OR_PROVINCE, 1.0));
    data.get(stanford).add(fill(stanford, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, "california", NERTag.STATE_OR_PROVINCE, 0.9));  // cut
    data.get(stanford).add(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl Â Blake", NERTag.PERSON, 1.0));
    data.get(stanford).add(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl Blake",  NERTag.PERSON, 0.9));  // cut
    data.get(stanford).add(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl  Blake", NERTag.PERSON, 0.9));  // cut
    data.get(stanford).add(fill(stanford, RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, "60,000", NERTag.NUMBER, 1.0));
    data.get(stanford).add(fill(stanford, RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, "\"60,000\"", NERTag.NUMBER, 0.9));  // cut
    data.get(stanford).add(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities", NERTag.ORGANIZATION, 1.0));
    data.get(stanford).add(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities LLC", NERTag.ORGANIZATION, 0.9));  // cut
    data.get(stanford).add(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard Madoff Investment Securities", NERTag.ORGANIZATION, 0.9));  // cut
    data.get(stanford).add(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Illinois Tool Works , Inc.", NERTag.ORGANIZATION, 1.0));
    data.get(stanford).add(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Illinois Tool Works of Glenville", NERTag.ORGANIZATION, 0.9));
    data.get(stanford).add(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "ITW", NERTag.ORGANIZATION, 0.8));


    return data;
  }

  protected static CoreMap mkCoreMap(String[] sentenceGloss, String[] posGloss, String[] nerGloss, Pair<Integer, String>... antecedents) {
    CoreMap sentence = new ArrayCoreMap(4);
    List<CoreLabel> tokens = new ArrayList<>(sentenceGloss.length);
    for (int i = 0; i < sentenceGloss.length; i++) {
      CoreLabel token = new CoreLabel();
      token.setWord(sentenceGloss[i]);
      token.setTag(posGloss[i]);
      token.setNER(nerGloss[i]);
      tokens.add(token);
    }
    for (Pair<Integer, String> pair : antecedents) {
      tokens.get(pair.first).set(CoreAnnotations.AntecedentAnnotation.class, pair.second);
    }
    sentence.set(CoreAnnotations.TokensAnnotation.class, tokens);
    return sentence;
  }

  @SuppressWarnings("unchecked")
  protected Map<KBPEntity, List<KBPSlotFill>> augmentedDummyData() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();

    // Build a coremap by hand!
    String[] sentenceGloss = "Executive NLPer Julie was born in Canada and lived in it where she founded KBP Incorporated on July 29 2013 .".split("\\s+");
    String[] posGloss = "NN NN NNP VBD VBN IN NNP CC VBD IN DT WRB PRP VBD NNP NNP IN NNP CD CD .".split("\\s+");
    String[] nerGloss = "O O PERSON O O O COUNTRY O O O O O O O O ORGANIZATION O DATE DATE DATE O".split("\\s+");
    CoreMap sentence = mkCoreMap(sentenceGloss, posGloss, nerGloss,
        Pair.makePair(2, "Julie"), Pair.makePair(12, "Julie"), Pair.makePair(10, "Canada"), Pair.makePair(17, "2013-07-29"));

    // Fill slots
    data.put(julie, new ArrayList<KBPSlotFill>());
    data.put(kbpinc, new ArrayList<KBPSlotFill>());
    // (sanity check)
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", NERTag.COUNTRY, 0.8, 2, 3, 6, 7, sentence));
    // (expand antecedent)
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRIES_OF_RESIDENCE, "Canada", NERTag.COUNTRY, 0.4, 2, 3, 6, 11, sentence));
    // (expand title)
    data.get(julie).add(fill(julie, RelationType.PER_TITLE, "NLPer", NERTag.TITLE, 0.4, 2, 3, 1, 2, sentence));
    // (rewrite top employees)
    data.get(kbpinc).add(fill(kbpinc,
        RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "she", NERTag.PERSON, 0.7, 14, 15, 12, 13, sentence));
    // (normalize date_founded)
    data.get(kbpinc).add(fill(kbpinc,
        RelationType.ORG_FOUNDED, "July 29 2013", NERTag.DATE, 0.6, 14, 15, 17, 20, sentence));
    // (expand subsidiary)

    return data;
  }

  @SuppressWarnings("unchecked")
  protected Map<KBPEntity, List<KBPSlotFill>> births() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<>();

    // Build a coremap by hand!
    String[] sentenceGloss = "Julie was born in Canada and lived in San Francisco where she was fascinated by Montana in December .".split("\\s+");
    String[] posGloss = "NNP VBD VBN IN NNP CC VBD IN NNP NNP WRB PRP VBD VBN IN NNP IN NNP .".split("\\s+");
    String[] nerGloss = "PERSON O O O COUNTRY O O O CITY CITY O O O O O STATE_OR_PROVINCE O DATE O".split("\\s+");
    CoreMap sentence = mkCoreMap(sentenceGloss, posGloss, nerGloss);

    sentenceGloss = "Julie was raised in Toronto .".split("\\s+");
    posGloss = "NNP VBD VBN IN NNP .".split("\\s+");
    nerGloss = "PERSON O O O CITY O".split("\\s+");
    CoreMap sentence2 = mkCoreMap(sentenceGloss, posGloss, nerGloss);

    sentenceGloss = "Julie has absolutely nothing to do with the slot value Mexico .".split("\\s+");
    posGloss = "NNP X X X X X X X X X NNP .".split("\\s+");
    nerGloss = "PERSON O O O O O O O O O COUNTRY O".split("\\s+");
    CoreMap sentence3 = mkCoreMap(sentenceGloss, posGloss, nerGloss);

    sentenceGloss = "Hamilton died last month .".split("\\s+");
    posGloss = "NNP VBD NN NN .".split("\\s+");
    nerGloss = "PERSON O DATE DATE O".split("\\s+");
    CoreMap sentence4 = mkCoreMap(sentenceGloss, posGloss, nerGloss);

    // Fill slots
    data.put(julie, new ArrayList<KBPSlotFill>());
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", NERTag.COUNTRY, 0.2, 0, 1, 4, 5, sentence));
    data.get(julie).add(fill(julie, RelationType.PER_CITY_OF_BIRTH, "San Francisco", NERTag.CITY, 0.4, 0, 1, 8, 10, sentence));
    data.get(julie).add(fill(julie, RelationType.PER_DATE_OF_BIRTH, "December", NERTag.DATE, 0.4, 0, 1, 17, 18, sentence));
    data.get(julie).add(fill(julie, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, "Montana", NERTag.STATE_OR_PROVINCE, 0.4, 0, 1, 15, 16, sentence));
    data.get(julie).add(fill(julie, RelationType.PER_CITY_OF_BIRTH, "Toronto", NERTag.CITY, 0.4, 0, 1, 4, 5, sentence2));
    data.get(julie).add(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Mexico", NERTag.CITY, 0.8, 0, 1, 10, 11, sentence3));

    data.put(hamilton, new ArrayList<KBPSlotFill>());
    data.get(hamilton).add(fill(hamilton, RelationType.PER_DATE_OF_DEATH, "last month", NERTag.DATE, 0.8, 0, 1, 2, 4, sentence4));

    return data;
  }
}
