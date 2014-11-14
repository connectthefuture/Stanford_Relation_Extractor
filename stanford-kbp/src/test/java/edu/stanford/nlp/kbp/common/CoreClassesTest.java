package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.util.MetaClass;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

/**
 * A test to ensure equality / hash code / etc for the core KBP
 * classes such as KBPOfficialEntity, KBPair, KBTriple, etc.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("Convert2Diamond")
public class CoreClassesTest {


  @Test
  public void testKBPEntity() {
    eq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity());
    eq( KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity() );
    eq( KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity(),
        KBPNew.entName("entityone").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity() );
    eq( KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity(),
        KBPNew.entName("entityone").entType(NERTag.ORGANIZATION).entId("E0001").KBPOfficialEntity() );

    neq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.ORGANIZATION).KBPOfficialEntity());
    neq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0002").KBPOfficialEntity() );
    neq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0002").KBPOfficialEntity() );
    neq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity(),
        KBPNew.entName("EntityTwo").entType(NERTag.PERSON).KBPOfficialEntity());

    canSerialize(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPEntity());
    canSerialize(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity());
    canSerialize(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("foo").KBPEntity());
    canSerialize(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("foo").KBPOfficialEntity());
    canSerialize(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("foo").queryId("bar").KBPOfficialEntity());
    canSerialize(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("foo").queryId("bar").KBPOfficialEntity());
  }

  @Test
  public void KBPEntityEqualitySemantics() {
    eq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity());
    eq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity());
    eq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPEntity());
    eq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("entone").KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPEntity());
    eq( KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("entone").KBPOfficialEntity());
  }

  @Test
  public void testKBPair() {
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair()  );
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityone").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair()  );
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("null").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("null").slotValue("slotFillOne").KBPair()  );
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("(null)").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("(null)").slotValue("slotFillOne").KBPair()  );
    eq( KBPNew.entName("entityOne").entType(NERTag.ORGANIZATION).entId("E0001").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair()  );
    eq( KBPNew.from(KBPNew.entName("entity").entType(NERTag.PERSON).KBPOfficialEntity()).slotValue("slotFill").KBPair(),
        KBPNew.from(KBPNew.entName("entity").entType(NERTag.PERSON).KBPOfficialEntity()).slotValue("slotFill").KBPair());
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.PERSON).KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.PERSON).KBPair()  );
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").KBPair()  );

    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("(null)").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair()  );
    neq(KBPNew.entName("entityOne").entType(NERTag.ORGANIZATION).entId("(null)").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("(null)").slotValue("slotFillOne").KBPair()  );
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0002").slotValue("slotFillOne").KBPair()  );
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillTwo").KBPair()  );
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.ORGANIZATION).slotValue("slotFillOne").KBPair()  );
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.PERSON).KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.ORGANIZATION).KBPair()  );
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.ORGANIZATION).KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.PERSON).KBPair()  );
    // note[gabor]: this is a somewhat arbitrary choice of semantics; the primary motivation is so that a KBPair can be
    //              assessed for equality based only on its surface String form.
    //              This is less silly than it sounds: e.g., Postgres would like to reliably key KBPairs on some serialized representation
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.PERSON).KBPair()  );

    canSerialize(KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair());
  }

  @Test
  public void testKBTriple() {
    eq( new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    eq( new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("E0001"), "entityone", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    eq( new KBTriple(Maybe.Just("null"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("null"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    eq( new KBTriple(Maybe.Just("(null)"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("(null)"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    eq( new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.ORGANIZATION, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );

    neq(new KBTriple(Maybe.Just("(null)"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    neq(new KBTriple(Maybe.Just("(null)"), "entityOne", NERTag.ORGANIZATION, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("(null)"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    neq(new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("E0002"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    neq(new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_COUNTRY_OF_BIRTH.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );
    neq(new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing()),
        new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillTwo", Maybe.<NERTag>Nothing())  );
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").rel("r").KBTriple(),
        KBPNew.entName("entityOne").entType(NERTag.ORGANIZATION).slotValue("slotFillOne").rel("r").KBTriple()  );
    neq(KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.PERSON).rel("r").KBTriple(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").slotType(NERTag.ORGANIZATION).rel("r").KBTriple()  );

    canSerialize(KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").rel("foo").KBTriple());
  }

  @Test
  public void testKBPSlotFill() {
    eq( KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill());
    eq( KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").score(0.3).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").score(0.5).KBPSlotFill());

    neq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill());
    neq(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_death").KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill());
    eq(KBPNew.entName("entityone").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill());

    // Check sorting; particularly, the corner cases
    List<KBPSlotFill> indices = Arrays.asList(
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill5").rel("per:date_of_birth").score(0.5).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill2").rel("per:date_of_birth").score(0.1).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFillX").rel("per:date_of_birth").score(0.42).KBPSlotFill(),
        KBPNew.entName("EntityOneLonger").entType(NERTag.PERSON).entId("eid*").slotValue("slotFillX").rel("per:date_of_birth").score(0.42).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFillY").rel("per:date_of_birth").score(0.42).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFillX longer").rel("per:date_of_birth").score(0.42).KBPSlotFill());
    assertEquals(indices.size(), new HashSet<>(indices).size());
    List<KBPSlotFill> sorted = new ArrayList<>(indices);
    Collections.sort(sorted);
    assertEquals(indices.get(0), sorted.get(0));
    assertEquals(indices.get(5), sorted.get(1));
    assertEquals(indices.get(3), sorted.get(2));
    assertEquals(indices.get(2), sorted.get(3));
    assertEquals(indices.get(4), sorted.get(4));
    assertEquals(indices.get(1), sorted.get(5));

    canSerialize(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill());
  }

  @Test
  public void testKBPSlotFillOrdering() {
    assertEquals(0, KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("shortString").rel("per:date_of_birth").score(0.5).KBPSlotFill().compareTo(
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("shortString").rel("per:date_of_birth").score(0.5).KBPSlotFill()
    ));

    // Test score tie-breaker
    KBPSlotFill[] fills = new KBPSlotFill[]{KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("shortString").rel("per:date_of_birth").score(0.5).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("shortString").rel("per:date_of_birth").score(0.75).KBPSlotFill()};
    Arrays.sort(fills);
    assertEquals(0.75, fills[0].score.orCrash(), 0.0);
    assertEquals(0.5, fills[1].score.orCrash(), 0.0);

    // Test length tie-breaker
    assertTrue(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("shortString").rel("per:date_of_birth").score(0.5).KBPSlotFill().compareTo(
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("longlonglonglonglongString").rel("per:date_of_birth").score(0.5).KBPSlotFill()
    ) > 0);
    fills = new KBPSlotFill[]{KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("shortString").rel("per:date_of_birth").score(0.5).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("(null)").slotValue("longlonglonglonglongString").rel("per:date_of_birth").score(0.5).KBPSlotFill()};
    Arrays.sort(fills);
    assertEquals("longlonglonglonglongString", fills[0].key.slotValue);
    assertEquals("shortString", fills[1].key.slotValue);
  }

  @Test
  public void testProperties() {
    // Make sure we can load a parameterized Maybe
    Maybe<File> target = MetaClass.cast("/tmp/", Maybe.class);
    assertNotNull(target);
    assertTrue(target.isDefined());
    assertEquals(new File("/tmp/"), target.get());


    Set<String> features = MetaClass.cast("{ ATLEAST_ONCE, COOC }", Set.class);
    assertNotNull(features);
    assertTrue(features.contains(Props.Y_FEATURE_CLASS.ATLEAST_ONCE.name()));
    assertTrue(features.contains(Props.Y_FEATURE_CLASS.COOC.name()));
    assertEquals(2, features.size());
  }

  @Test
  public void testSpan() {
    Span span = new Span(10, 15);
    assertTrue(span.iterator().hasNext());
    assertEquals(10, (int) span.iterator().next());
    int last = -1;
    for (int i : span) {
      last = i;
    }
    assertEquals(14, last);
  }

  @Test
  public void testRelationTypeFromString() {
    assertEquals(RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, RelationType.fromString("org:number_of_employees/members").orCrash());
    assertEquals(RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, RelationType.fromString("org:number_of_employeesSLASHmembers").orCrash());
  }

  @Test
  public void testKBPNewConstruct() {
    // Entity
    eq( KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPEntity(),
        new KBPEntity("EntityOne", NERTag.PERSON) );

    // Official Entity
    eq( KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").KBPOfficialEntity(),
        KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").KBPOfficialEntity() );
    neq(KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").queryId("foo").KBPOfficialEntity(),
        KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").KBPOfficialEntity() );

    // KBPair
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair()  );

    // KBTriple
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").rel(RelationType.PER_AGE).KBTriple(),
        new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );

  }

  @Test
  public void testKBPNewEquality() {
    // Entity
    eq( KBPNew.from(new KBPEntity("EntityOne", NERTag.PERSON)).KBPEntity(),
        new KBPEntity("EntityOne", NERTag.PERSON) );
    eq( KBPNew.from(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity()).KBPEntity(),
        new KBPEntity("EntityOne", NERTag.PERSON) );
    eq( KBPNew.from( KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").KBPOfficialEntity() ).KBPEntity(),
        new KBPEntity("EntityTwo", NERTag.ORGANIZATION) );
    neq(KBPNew.from( KBPNew.entName("EntityTwo").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity() ).KBPEntity(),
        new KBPEntity("EntityTwo", NERTag.ORGANIZATION) );

    // Official entity
    eq(KBPNew.from(KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity()).KBPOfficialEntity(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).KBPOfficialEntity());
    eq( KBPNew.from( KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity() ).KBPOfficialEntity(),
        KBPNew.entName("entityone").entType(NERTag.ORGANIZATION).entId("E0001").KBPOfficialEntity() );
    neq(KBPNew.from( KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("E0001").KBPOfficialEntity() ).KBPOfficialEntity(),
        new KBPEntity("EntityTwo", NERTag.ORGANIZATION) );

    // KBPair
    eq( KBPNew.from(KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair()).KBPair(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").slotValue("slotFillOne").KBPair()  );

    // KBTriple
    eq( KBPNew.from(new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())).KBTriple(),
        new KBTriple(Maybe.Just("E0001"), "entityOne", NERTag.PERSON, RelationType.PER_AGE.canonicalName, "slotFillOne", Maybe.<NERTag>Nothing())  );

    // KBPSlotFill
    eq( KBPNew.from(KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill()).KBPSlotFill(),
        KBPNew.entName("EntityOne").entType(NERTag.PERSON).entId("eid").slotValue("slotFill1").rel("per:date_of_birth").KBPSlotFill());
  }

  @Test
  public void testKBPNewRewriteEntity() {
    // Create Entity
    KBPEntity entity = KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).KBPEntity();
    assertEquals("EntityTwo", entity.name);
    assertEquals(NERTag.ORGANIZATION, entity.type);

    // Entity name
    assertEquals("EntityOne", KBPNew.from(entity).entName("EntityOne").KBPEntity().name);
    eq( KBPNew.from(entity).entName("EntityTwo").KBPEntity(), entity);
    // Entity type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(entity).entType(NERTag.ORGANIZATION).KBPEntity().type);
    eq( KBPNew.from(entity).entType(NERTag.ORGANIZATION).KBPEntity(), entity);
  }

  @Test
  public void testKBPNewRewriteOfficialEntity() {
    // Create official entity
    KBPOfficialEntity officialEntity = KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").queryId("qid1").ignoredSlots(new HashSet<RelationType>()).KBPOfficialEntity();
    assertEquals("EntityTwo", officialEntity.name);
    assertEquals(NERTag.ORGANIZATION, officialEntity.type);
    assertEquals("E0001", officialEntity.id.orCrash());
    assertEquals("qid1", officialEntity.queryId.orCrash());
    assertEquals(new HashSet<RelationType>(), officialEntity.ignoredSlots.orCrash());
    // Entity name
    assertEquals("EntityOne", KBPNew.from(officialEntity).entName("EntityOne").KBPOfficialEntity().name);
    eq( KBPNew.from(officialEntity).entName("EntityTwo").KBPOfficialEntity(), officialEntity);
    // Entity type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(officialEntity).entType(NERTag.ORGANIZATION).KBPOfficialEntity().type);
    eq( KBPNew.from(officialEntity).entType(NERTag.ORGANIZATION).KBPOfficialEntity(), officialEntity);
    // Entity id
    assertEquals("E0002", KBPNew.from(officialEntity).entId("E0002").KBPOfficialEntity().id.orCrash());
    assertEquals("E0002", KBPNew.from(officialEntity).entId(Maybe.Just("E0002")).KBPOfficialEntity().id.orCrash());
    assertFalse(KBPNew.from(officialEntity).entId(Maybe.<String>Nothing()).KBPOfficialEntity().id.isDefined());
    eq(KBPNew.from(officialEntity).entId("E0001").KBPOfficialEntity(), officialEntity);
    eq( KBPNew.from(officialEntity).entId(Maybe.Just("E0001")).KBPOfficialEntity(), officialEntity );
    // Query id
    assertEquals("qid2", KBPNew.from(officialEntity).queryId("qid2").KBPOfficialEntity().queryId.orCrash());
    assertEquals("qid2", KBPNew.from(officialEntity).queryId(Maybe.Just("qid2")).KBPOfficialEntity().queryId.orCrash());
    assertEquals("E0001", KBPNew.from(officialEntity).queryId(Maybe.Just("qid2")).KBPOfficialEntity().id.orCrash());
    assertFalse(KBPNew.from(officialEntity).queryId(Maybe.<String>Nothing()).KBPOfficialEntity().queryId.isDefined());
    eq( KBPNew.from(officialEntity).queryId("qid1").KBPOfficialEntity(), officialEntity );
    eq( KBPNew.from(officialEntity).queryId(Maybe.Just("qid1")).KBPOfficialEntity(), officialEntity );
    // Ignored slots
    Set<RelationType> ignored = new HashSet<RelationType>(){{ add(RelationType.ORG_ALTERNATE_NAMES); }};
    assertEquals(ignored, KBPNew.from(officialEntity).ignoredSlots(ignored).KBPOfficialEntity().ignoredSlots.orCrash());
    assertEquals(ignored, KBPNew.from(officialEntity).ignoredSlots(Maybe.Just(ignored)).KBPOfficialEntity().ignoredSlots.orCrash());
    assertEquals("E0001", KBPNew.from(officialEntity).ignoredSlots(Maybe.Just(ignored)).KBPOfficialEntity().id.orCrash());
    assertFalse(KBPNew.from(officialEntity).ignoredSlots(Maybe.<Set<RelationType>>Nothing()).KBPOfficialEntity().ignoredSlots.isDefined());
    eq( KBPNew.from(officialEntity).ignoredSlots(new HashSet<RelationType>()).KBPOfficialEntity(), officialEntity );
    eq( KBPNew.from(officialEntity).ignoredSlots(Maybe.Just((Set<RelationType>) new HashSet<RelationType>())).KBPOfficialEntity(), officialEntity );
  }

  @Test
  public void testKBPNewRewriteKBPair() {
    // Create official entity
    KBPair kbPair = KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").queryId("qid1").ignoredSlots(new HashSet<RelationType>()).slotValue("value1").slotType(NERTag.PERSON).KBPair();
    assertEquals("EntityTwo", kbPair.entityName);
    assertEquals(NERTag.ORGANIZATION, kbPair.entityType);
    assertEquals("E0001", kbPair.entityId.orCrash());
    assertEquals("value1", kbPair.slotValue);
    assertEquals(NERTag.PERSON, kbPair.slotType.orCrash());
    // Entity name
    assertEquals("EntityOne", KBPNew.from(kbPair).entName("EntityOne").KBPair().entityName);
    eq( KBPNew.from(kbPair).entName("EntityTwo").KBPair(), kbPair);
    // Entity type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(kbPair).entType(NERTag.ORGANIZATION).KBPair().entityType);
    eq( KBPNew.from(kbPair).entType(NERTag.ORGANIZATION).KBPair(), kbPair);
    // Entity id
    assertEquals("E0002", KBPNew.from(kbPair).entId("E0002").KBPair().entityId.orCrash());
    assertEquals("E0002", KBPNew.from(kbPair).entId(Maybe.Just("E0002")).KBPair().entityId.orCrash());
    eq( KBPNew.from(kbPair).entId("E0001").KBPair(), kbPair );
    eq( KBPNew.from(kbPair).entId(Maybe.Just("E0001")).KBPair(), kbPair );
    // Query id
    assertEquals("qid2", KBPNew.from(kbPair).queryId("qid2").KBPOfficialEntity().queryId.orCrash());
    assertEquals("qid2", KBPNew.from(kbPair).queryId(Maybe.Just("qid2")).KBPOfficialEntity().queryId.orCrash());
    assertEquals("E0001", KBPNew.from(kbPair).queryId(Maybe.Just("qid2")).KBPair().entityId.orCrash());
    assertFalse(KBPNew.from(kbPair).queryId(Maybe.<String>Nothing()).KBPOfficialEntity().queryId.isDefined());
    eq(KBPNew.from(kbPair).queryId("qid1").KBPair(), kbPair);
    eq( KBPNew.from(kbPair).queryId(Maybe.Just("qid1")).KBPair(), kbPair );
    // Ignored slots
    Set<RelationType> ignored = new HashSet<RelationType>(){{ add(RelationType.ORG_ALTERNATE_NAMES); }};
    assertEquals(ignored, KBPNew.from(kbPair).ignoredSlots(ignored).KBPOfficialEntity().ignoredSlots.orCrash());
    assertEquals(ignored, KBPNew.from(kbPair).ignoredSlots(Maybe.Just(ignored)).KBPOfficialEntity().ignoredSlots.orCrash());
    assertFalse(KBPNew.from(kbPair).ignoredSlots(Maybe.<Set<RelationType>>Nothing()).KBPOfficialEntity().ignoredSlots.isDefined());
    assertEquals("E0001", KBPNew.from(kbPair).ignoredSlots(Maybe.Just(ignored)).KBPair().entityId.orCrash());
    eq( KBPNew.from(kbPair).ignoredSlots(new HashSet<RelationType>()).KBPair(), kbPair );
    eq( KBPNew.from(kbPair).ignoredSlots(Maybe.Just((Set<RelationType>) new HashSet<RelationType>())).KBPair(), kbPair );
    // Slot Value
    assertEquals("value2", KBPNew.from(kbPair).slotValue("value2").KBPair().slotValue);
    assertEquals("E0001", KBPNew.from(kbPair).slotValue("value2").KBPair().entityId.orCrash());
    eq( KBPNew.from(kbPair).slotValue("value1").KBPair(), kbPair );
    // Slot Type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(kbPair).slotType(NERTag.ORGANIZATION).KBPair().slotType.orCrash());
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(kbPair).slotType(Maybe.Just(NERTag.ORGANIZATION)).KBPair().slotType.orCrash());
    assertFalse(KBPNew.from(kbPair).slotType(Maybe.<NERTag>Nothing()).KBPair().slotType.isDefined());
    assertEquals("E0001", KBPNew.from(kbPair).slotType(NERTag.ORGANIZATION).KBPair().entityId.orCrash());
    assertEquals("E0001", KBPNew.from(kbPair).slotType(Maybe.Just(NERTag.ORGANIZATION)).KBPair().entityId.orCrash());
    eq(KBPNew.from(kbPair).slotType(NERTag.PERSON).KBPair(), kbPair);
    eq( KBPNew.from(kbPair).slotType(Maybe.Just(NERTag.PERSON)).KBPair(), kbPair );
    // Entire entity
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").ent(KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).KBPEntity()).KBPair(),
        KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).slotValue("slotFillOne").KBPair()  );
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").ent(KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).entId("42").KBPOfficialEntity()).KBPair(),
        KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).slotValue("slotFillOne").entId("42").KBPair()  );
  }

  @Test
  public void testKBPNewRewriteKBTriple() {
    // Create official entity
    KBTriple kbTriple = KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").queryId("qid1").ignoredSlots(new HashSet<RelationType>()).slotValue("value1").slotType(NERTag.PERSON).rel(RelationType.PER_ORIGIN).KBTriple();
    assertEquals("EntityTwo", kbTriple.entityName);
    assertEquals(NERTag.ORGANIZATION, kbTriple.entityType);
    assertEquals("E0001", kbTriple.entityId.orCrash());
    assertEquals("value1", kbTriple.slotValue);
    assertEquals(NERTag.PERSON, kbTriple.slotType.orCrash());
    assertEquals(RelationType.PER_ORIGIN.canonicalName, kbTriple.relationName);
    // Entity name
    assertEquals("EntityOne", KBPNew.from(kbTriple).entName("EntityOne").KBTriple().entityName);
    eq( KBPNew.from(kbTriple).entName("EntityTwo").KBTriple(), kbTriple);
    // Entity type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(kbTriple).entType(NERTag.ORGANIZATION).KBTriple().entityType);
    eq( KBPNew.from(kbTriple).entType(NERTag.ORGANIZATION).KBTriple(), kbTriple);
    // Entity id
    assertEquals("E0002", KBPNew.from(kbTriple).entId("E0002").KBTriple().entityId.orCrash());
    assertEquals("E0002", KBPNew.from(kbTriple).entId(Maybe.Just("E0002")).KBTriple().entityId.orCrash());
    eq( KBPNew.from(kbTriple).entId("E0001").KBTriple(), kbTriple );
    eq( KBPNew.from(kbTriple).entId(Maybe.Just("E0001")).KBTriple(), kbTriple );
    // Query id
    assertEquals("qid2", KBPNew.from(kbTriple).queryId("qid2").KBPOfficialEntity().queryId.orCrash());
    assertEquals("qid2", KBPNew.from(kbTriple).queryId(Maybe.Just("qid2")).KBPOfficialEntity().queryId.orCrash());
    assertFalse(KBPNew.from(kbTriple).queryId(Maybe.<String>Nothing()).KBPOfficialEntity().queryId.isDefined());
    assertEquals("E0001", KBPNew.from(kbTriple).queryId(Maybe.Just("qid2")).KBTriple().entityId.orCrash());
    eq( KBPNew.from(kbTriple).queryId("qid1").KBTriple(), kbTriple );
    eq( KBPNew.from(kbTriple).queryId(Maybe.Just("qid1")).KBTriple(), kbTriple );
    // Ignored slots
    Set<RelationType> ignored = new HashSet<RelationType>(){{ add(RelationType.ORG_ALTERNATE_NAMES); }};
    assertEquals(ignored, KBPNew.from(kbTriple).ignoredSlots(ignored).KBPOfficialEntity().ignoredSlots.orCrash());
    assertEquals(ignored, KBPNew.from(kbTriple).ignoredSlots(Maybe.Just(ignored)).KBPOfficialEntity().ignoredSlots.orCrash());
    assertFalse(KBPNew.from(kbTriple).ignoredSlots(Maybe.<Set<RelationType>>Nothing()).KBPOfficialEntity().ignoredSlots.isDefined());
    assertEquals("E0001", KBPNew.from(kbTriple).ignoredSlots(Maybe.Just(ignored)).KBTriple().entityId.orCrash());
    eq( KBPNew.from(kbTriple).ignoredSlots(new HashSet<RelationType>()).KBTriple(), kbTriple );
    eq( KBPNew.from(kbTriple).ignoredSlots(Maybe.Just((Set<RelationType>) new HashSet<RelationType>())).KBTriple(), kbTriple );
    // Slot Value
    assertEquals("value2", KBPNew.from(kbTriple).slotValue("value2").KBTriple().slotValue);
    assertEquals("E0001", KBPNew.from(kbTriple).slotValue("value2").KBTriple().entityId.orCrash());
    eq( KBPNew.from(kbTriple).slotValue("value1").KBTriple(), kbTriple );
    // Slot Type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(kbTriple).slotType(NERTag.ORGANIZATION).KBTriple().slotType.orCrash());
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(kbTriple).slotType(Maybe.Just(NERTag.ORGANIZATION)).KBTriple().slotType.orCrash());
    assertFalse(KBPNew.from(kbTriple).slotType(Maybe.<NERTag>Nothing()).KBPair().slotType.isDefined());
    assertEquals("E0001", KBPNew.from(kbTriple).slotType(NERTag.ORGANIZATION).KBTriple().entityId.orCrash());
    assertEquals("E0001", KBPNew.from(kbTriple).slotType(Maybe.Just(NERTag.ORGANIZATION)).KBTriple().entityId.orCrash());
    eq( KBPNew.from(kbTriple).slotType(NERTag.PERSON).KBTriple(), kbTriple );
    eq( KBPNew.from(kbTriple).slotType(Maybe.Just(NERTag.PERSON)).KBTriple(), kbTriple );
    // Relation
    assertEquals(RelationType.ORG_FOUNDED.canonicalName, KBPNew.from(kbTriple).rel(RelationType.ORG_FOUNDED).KBTriple().relationName);
    assertEquals(RelationType.ORG_FOUNDED.canonicalName, KBPNew.from(kbTriple).rel(RelationType.ORG_FOUNDED.canonicalName).KBTriple().relationName);
    assertEquals("E0001", KBPNew.from(kbTriple).rel(RelationType.ORG_FOUNDED).KBTriple().entityId.orCrash());
    assertEquals("E0001", KBPNew.from(kbTriple).rel(RelationType.ORG_FOUNDED.canonicalName).KBTriple().entityId.orCrash());
    eq( KBPNew.from(kbTriple).rel(RelationType.PER_ORIGIN).KBTriple(), kbTriple );
    // Overwrite entity
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").ent(KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).KBPEntity()).rel("foo").KBTriple(),
        KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).slotValue("slotFillOne").rel("foo").KBTriple()  );
    eq( KBPNew.entName("entityOne").entType(NERTag.PERSON).slotValue("slotFillOne").ent(KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).entId("42").KBPOfficialEntity()).rel("foo").KBTriple(),
        KBPNew.entName("entityTwo").entType(NERTag.ORGANIZATION).slotValue("slotFillOne").entId("42").rel("foo").KBTriple()  );
  }

  @Test
  public void testKBPNewRewriteKBPSlotFill() {
    KBPRelationProvenance provA = new KBPRelationProvenance("foo", "bar");
    KBPRelationProvenance provB = new KBPRelationProvenance("foo2", "bar2");
    // Create official entity
    KBPSlotFill slotFill = KBPNew.entName("EntityTwo").entType(NERTag.ORGANIZATION).entId("E0001").queryId("qid1").ignoredSlots(new HashSet<RelationType>()).slotValue("value1").slotType(NERTag.PERSON).rel(RelationType.PER_ORIGIN).provenance(provA).score(1.0).KBPSlotFill();
    assertEquals("EntityTwo", slotFill.key.entityName);
    assertEquals(NERTag.ORGANIZATION, slotFill.key.entityType);
    assertEquals("E0001", slotFill.key.entityId.orCrash());
    assertEquals("value1", slotFill.key.slotValue);
    assertEquals(NERTag.PERSON, slotFill.key.slotType.orCrash());
    assertEquals(RelationType.PER_ORIGIN.canonicalName, slotFill.key.relationName);
    // Entity name
    assertEquals("EntityOne", KBPNew.from(slotFill).entName("EntityOne").KBPSlotFill().key.entityName);
    eq( KBPNew.from(slotFill).entName("EntityTwo").KBPSlotFill(), slotFill);
    // Entity type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(slotFill).entType(NERTag.ORGANIZATION).KBPSlotFill().key.entityType);
    eq( KBPNew.from(slotFill).entType(NERTag.ORGANIZATION).KBPSlotFill(), slotFill);
    // Entity id
    assertEquals("E0002", KBPNew.from(slotFill).entId("E0002").KBPSlotFill().key.entityId.orCrash());
    assertEquals("E0002", KBPNew.from(slotFill).entId(Maybe.Just("E0002")).KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).entId("E0001").KBPSlotFill(), slotFill );
    eq( KBPNew.from(slotFill).entId(Maybe.Just("E0001")).KBPSlotFill(), slotFill );
    // Query id
    assertEquals("qid2", KBPNew.from(slotFill).queryId("qid2").KBPOfficialEntity().queryId.orCrash());
    assertEquals("qid2", KBPNew.from(slotFill).queryId(Maybe.Just("qid2")).KBPOfficialEntity().queryId.orCrash());
    assertFalse(KBPNew.from(slotFill).queryId(Maybe.<String>Nothing()).KBPOfficialEntity().queryId.isDefined());
    assertEquals("E0001", KBPNew.from(slotFill).queryId(Maybe.Just("qid2")).KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).queryId("qid1").KBPSlotFill(), slotFill );
    eq( KBPNew.from(slotFill).queryId(Maybe.Just("qid1")).KBPSlotFill(), slotFill );
    // Ignored slots
    Set<RelationType> ignored = new HashSet<RelationType>(){{ add(RelationType.ORG_ALTERNATE_NAMES); }};
    assertEquals(ignored, KBPNew.from(slotFill).ignoredSlots(ignored).KBPOfficialEntity().ignoredSlots.orCrash());
    assertEquals(ignored, KBPNew.from(slotFill).ignoredSlots(Maybe.Just(ignored)).KBPOfficialEntity().ignoredSlots.orCrash());
    assertFalse(KBPNew.from(slotFill).ignoredSlots(Maybe.<Set<RelationType>>Nothing()).KBPOfficialEntity().ignoredSlots.isDefined());
    assertEquals("E0001", KBPNew.from(slotFill).ignoredSlots(Maybe.Just(ignored)).KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).ignoredSlots(new HashSet<RelationType>()).KBPSlotFill(), slotFill );
    eq( KBPNew.from(slotFill).ignoredSlots(Maybe.Just((Set<RelationType>) new HashSet<RelationType>())).KBPSlotFill(), slotFill );
    // Slot Value
    assertEquals("value2", KBPNew.from(slotFill).slotValue("value2").KBPSlotFill().key.slotValue);
    assertEquals("E0001", KBPNew.from(slotFill).slotValue("value2").KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).slotValue("value1").KBPSlotFill(), slotFill );
    // Slot Type
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(slotFill).slotType(NERTag.ORGANIZATION).KBPSlotFill().key.slotType.orCrash());
    assertEquals(NERTag.ORGANIZATION, KBPNew.from(slotFill).slotType(Maybe.Just(NERTag.ORGANIZATION)).KBPSlotFill().key.slotType.orCrash());
    assertFalse(KBPNew.from(slotFill).slotType(Maybe.<NERTag>Nothing()).KBPair().slotType.isDefined());
    assertEquals("E0001", KBPNew.from(slotFill).slotType(NERTag.ORGANIZATION).KBPSlotFill().key.entityId.orCrash());
    assertEquals("E0001", KBPNew.from(slotFill).slotType(Maybe.Just(NERTag.ORGANIZATION)).KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).slotType(NERTag.PERSON).KBPSlotFill(), slotFill );
    eq( KBPNew.from(slotFill).slotType(Maybe.Just(NERTag.PERSON)).KBPSlotFill(), slotFill );
    // Relation
    assertEquals(RelationType.ORG_FOUNDED.canonicalName, KBPNew.from(slotFill).rel(RelationType.ORG_FOUNDED).KBPSlotFill().key.relationName);
    assertEquals(RelationType.ORG_FOUNDED.canonicalName, KBPNew.from(slotFill).rel(RelationType.ORG_FOUNDED.canonicalName).KBPSlotFill().key.relationName);
    assertEquals("E0001", KBPNew.from(slotFill).rel(RelationType.ORG_FOUNDED).KBPSlotFill().key.entityId.orCrash());
    assertEquals("E0001", KBPNew.from(slotFill).rel(RelationType.ORG_FOUNDED.canonicalName).KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).rel(RelationType.PER_ORIGIN).KBPSlotFill(), slotFill );
    // Provenance
    assertEquals(provB, KBPNew.from(slotFill).provenance(provB).KBPSlotFill().provenance.orCrash());
    assertEquals(provB, KBPNew.from(slotFill).provenance(Maybe.Just(provB)).KBPSlotFill().provenance.orCrash());
    assertFalse(KBPNew.from(slotFill).provenance(Maybe.<KBPRelationProvenance>Nothing()).KBPSlotFill().provenance.isDefined());
    assertEquals("E0001", KBPNew.from(slotFill).provenance(provB).KBPSlotFill().key.entityId.orCrash());
    assertEquals("E0001", KBPNew.from(slotFill).provenance(Maybe.Just(provB)).KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).provenance(provA).KBPSlotFill(), slotFill );
    eq( KBPNew.from(slotFill).provenance(Maybe.Just(provA)).KBPSlotFill(), slotFill );
    // Score
    assertEquals(0.0, KBPNew.from(slotFill).score(0.0).KBPSlotFill().score.orCrash(), 1e-5);
    assertEquals(0.0, KBPNew.from(slotFill).score(Maybe.Just(0.0)).KBPSlotFill().score.orCrash(), 1e-5);
    assertFalse(KBPNew.from(slotFill).score(Maybe.<Double>Nothing()).KBPSlotFill().score.isDefined());
    assertEquals("E0001", KBPNew.from(slotFill).score(0.0).KBPSlotFill().key.entityId.orCrash());
    assertEquals("E0001", KBPNew.from(slotFill).score(Maybe.Just(0.0)).KBPSlotFill().key.entityId.orCrash());
    eq( KBPNew.from(slotFill).score(1.0).KBPSlotFill(), slotFill );
    eq( KBPNew.from(slotFill).score(Maybe.Just(1.0)).KBPSlotFill(), slotFill );
  }

  @Test
  public void testKBPNewUseCases() {
    eq( KBPNew.from(
          KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").queryId("queryFoo").slotValue("slotFillOne").KBPair()
        ).KBPOfficialEntity(),
        KBPNew.entName("entityOne").entType(NERTag.PERSON).entId("E0001").queryId("queryFoo").KBPOfficialEntity());
  }

  @Test
  public void testNERTagShortName() {
    for (NERTag tag : NERTag.values()) {
      assertEquals(tag, NERTag.fromShortName(tag.shortName).orCrash());
    }
  }

  private static void canSerialize(KBPEntity entity) {
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    try {
      entity.toProto().writeTo(os);
      os.close();
      assertEquals(entity, KBPNew.from(KBPProtos.KBPEntity.parseFrom(os.toByteArray())).KBPEntity());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void canSerialize(KBPair pair) {
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    try {
      pair.toProto().writeTo(os);
      os.close();
      assertEquals(pair, KBPNew.from(KBPProtos.KBTuple.parseFrom(os.toByteArray())).KBPair());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void canSerialize(KBTriple triple) {
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    try {
      triple.toProto().writeTo(os);
      os.close();
      assertEquals(triple, KBPNew.from(KBPProtos.KBTuple.parseFrom(os.toByteArray())).KBTriple());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void canSerialize(KBPSlotFill fill) {
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    try {
      fill.toProto().writeTo(os);
      os.close();
      assertEquals(fill, KBPNew.from(KBPProtos.KBPSlotFill.parseFrom(os.toByteArray())).KBPSlotFill());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


  private static void eq(Object a, Object b) {
    assertEquals(a, b);
    assertEquals(a.hashCode(), b.hashCode());
  }

  private static void neq(Object a, Object b) {
    assertNotSame(a, b);
  }
}
