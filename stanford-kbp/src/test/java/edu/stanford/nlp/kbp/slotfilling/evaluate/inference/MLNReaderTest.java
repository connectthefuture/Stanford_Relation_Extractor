package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;

import org.junit.Ignore;
import org.junit.Test;

import static edu.stanford.nlp.util.logging.Redwood.Util.log;

/**
 * Test for MLNReader
 */
@Ignore
public class MLNReaderTest {

  @Test
  public void testMLNParsing() {
    String basicMLNText = new MLNTextBuilder()
            .addPredicates()
            .openPredicate("LivesIn", "person", "place")
            .openPredicate("WorksAt", "person", "org")
            .openPredicate("HeadquarteredAt", "org", "place")
            .endPredicates()
            .addRules()
            .newRule("unique place").orNot("HeadquarteredAt", "person", "place1" ).orNot("HeadquarteredAt", "person", "place2" ).equals("place1", "place2" ).endRule()
            .endRules()
            .end().toString();
    MLNText txt = MLNReader.parse(basicMLNText);
    log(txt.toString());
  }

  @Test
  public void testMLNParsing2() {
    String text =
"meet_up_with_TYPE_ORG_TO_PER(ORGANIZATION,PERSON)\n"+
"sail_to_TYPE_PER_TO_CRY(PERSON,COUNTRY)\n"+
"bring_TYPE_PER_TO_PER(PERSON,PERSON)\n"+
"pardon_TYPE_PER_TO_PER(PERSON,PERSON)\n"+
"be_TYPE_ST_TO_ORG(STATE_OR_PROVINCE,ORGANIZATION)\n"+
"be_queen_consort_of_TYPE_PER_TO_CRY(PERSON,COUNTRY)\n"+
"own_subsidiary_of_TYPE_ORG_TO_ORG(ORGANIZATION,ORGANIZATION)\n"+
"forbid_TYPE_PER_TO_PER(PERSON,PERSON)\n"+
"work_in_TYPE_PER_TO_CIT(PERSON,CITY)\n"+
"accept_deal_with_TYPE_ORG_TO_PER(ORGANIZATION,PERSON)\n"+
"begin_service_at_TYPE_ORG_TO_CIT(ORGANIZATION,CITY)\n"+
"\n"+
"-2.98893  !per_date_of_death_TYPE_PER_TO_DT(x3,x2) v !org_founded_TYPE_ORG_TO_DT(x0,x2) v !per_member_of_TYPE_PER_TO_ORG(x3,x1) v org_alternate_names_TYPE_ORG_TO_ORG(x0,x1)\n"+
"0.96281  !per_date_of_death_TYPE_PER_TO_DT(x3,x2) v !org_alternate_names_TYPE_ORG_TO_ORG(x0,x1) v !per_member_of_TYPE_PER_TO_ORG(x3,x1) v org_founded_TYPE_ORG_TO_DT(x0,x2)\n"+
"-1.08288  !org_founded_TYPE_ORG_TO_DT(x0,x2) v !org_alternate_names_TYPE_ORG_TO_ORG(x0,x1) v !per_member_of_TYPE_PER_TO_ORG(x3,x1) v per_date_of_death_TYPE_PER_TO_DT(x3,x2)\n"+
"-0.12779  !per_date_of_death_TYPE_PER_TO_DT(x3,x2) v !org_founded_TYPE_ORG_TO_DT(x0,x2) v !org_alternate_names_TYPE_ORG_TO_ORG(x0,x1) v per_member_of_TYPE_PER_TO_ORG(x3,x1)\n"+
"-1.18671  !org_alternate_names_TYPE_ORG_TO_ORG(x0,x1) v !org_founded_TYPE_ORG_TO_DT(x1,x2) v !per_member_of_TYPE_PER_TO_ORG(x3,x0) v per_date_of_death_TYPE_PER_TO_DT(x3,x2)\n"+
"-3.47290  !per_date_of_death_TYPE_PER_TO_DT(x3,x2) v !org_founded_TYPE_ORG_TO_DT(x1,x2) v !per_member_of_TYPE_PER_TO_ORG(x3,x0) v org_alternate_names_TYPE_ORG_TO_ORG(x0,x1)\n"+
"-0.43616  !per_date_of_death_TYPE_PER_TO_DT(x3,x2) v !org_alternate_names_TYPE_ORG_TO_ORG(x0,x1) v !per_member_of_TYPE_PER_TO_ORG(x3,x0) v org_founded_TYPE_ORG_TO_DT(x1,x2)\n"+
"0.05631  !per_date_of_death_TYPE_PER_TO_DT(x3,x2) v !org_alternate_names_TYPE_ORG_TO_ORG(x0,x1) v !org_founded_TYPE_ORG_TO_DT(x1,x2) v per_member_of_TYPE_PER_TO_ORG(x3,x0)\n";

    MLNText txt = MLNReader.parse(text);
    log(txt.toString());
  }

}
