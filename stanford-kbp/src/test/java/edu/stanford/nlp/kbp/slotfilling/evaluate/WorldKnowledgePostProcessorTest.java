package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.Maybe;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static junit.framework.Assert.*;

/**
 * A unit test for the world knowledge post processor.
 * For example, testing that the correct inferences are made for
 * ambiguous entries (e.g., San Francisco).
 *
 * @author Gabor Angeli
 */
public class WorldKnowledgePostProcessorTest {
  private final static String cities =
  "san francisco	ca	us	732072\n" +
	"san francisco	nm	us	7\n" +
	"san francisco plaza	nm	us	5\n" +
	"san francisco	tx	us	200\n" +
	"san francisco de asis	14	ar	34\n" +
	"san francisco de bellocq	01	ar	362\n" +
	"san francisco de chanar	05	ar	547\n" +
	"san francisco de laishi	09	ar	5456\n" +
	"san francisco del chanar	05	ar	4562\n" +
	"san francisco del monte de oro	19	ar	2345\n" +
	"san francisco de santa fe	21	ar	76\n" +
	"san francisco	02	ar	3456\n" +
	"san francisco	05	ar	3467\n" +
	"san gabriel	ca	us	7889\n" +
	"san geronimo	ca	us	98\n" +
	"san gregorio	ca	us	7689\n" +
	"san ignacio	ca	us	678\n" +
	"san jacinto	ca	us	4567\n" +
	"new york	ny	us	8107916\n";

  private final static String regions =
	"us	ca	california\n" +
	"us	nm	new mexico\n" +
	"us	tx	texas\n" +
	"us	14	rocha\n" +
	"ar	01	andijon\n" +
	"ar	05	khorazm\n" +
	"ar	09	qoraqalpoghiston\n" +
	"ar	19	sucre\n" +
	"ar	21	trujillo\n" +
	"ar	02	aiga-i-le-tai\n" +
	"ar	21	al jawf\n" +
	"ar	02	kwazulu-natal\n" +
	"ar	05	eastern cape\n" +
	"us	ny	new york\n";

  private final static String countries =
  "us	united states\n" +
  "ar	argentina";

  private final static String alternateNames =
  "united states	United States of America	America	the States	US	U.S.	USA	Usa	U.S.A.	Columbia	Freedonia	Appalachia	Alleghany	Usonia	Usona\n" +
  "argentina	Argentine Republic	la Argentina	the Argentine	Argentine Nation	United Provinces of the Río de la Plata	Argentine Confederation";

  private final static String abbreviation2city =
      "SF	San Francisco";

  private final static String code2nationality =
      "us	american\n" +
      "ar	argentinian";

  private WorldKnowledgePostProcessor processor;

  @Before
  public void setUp() {
    try {
      File dir = File.createTempFile("kbp_worldknowledge", ".dir");
      assertTrue("Could not delete temporary file to make way for directory", dir.delete());
      assertTrue("Could not create temporary directory", dir.mkdirs());
      IOUtils.writeStringToFile(countries, dir.getPath() + File.separator + "kbp_code2country.tab", "UTF-8");
      IOUtils.writeStringToFile(regions, dir.getPath() + File.separator + "kbp_code2region.tab", "UTF-8");
      IOUtils.writeStringToFile(cities, dir.getPath() + File.separator + "kbp_cities.tab", "UTF-8");
      IOUtils.writeStringToFile(alternateNames, dir.getPath() + File.separator + "kbp_alternate_country_names.tab", "UTF-8");
      IOUtils.writeStringToFile(abbreviation2city, dir.getPath() + File.separator + "kbp_abbreviation2city.tab", "UTF-8");
      IOUtils.writeStringToFile(code2nationality, dir.getPath() + File.separator + "kbp_countrycode2nationality.tab", "UTF-8");
      this.processor = new WorldKnowledgePostProcessor(dir);
    } catch (IOException e) {
      e.printStackTrace();
      assertTrue("Could not initialize a temporary directory", false);
    }
  }

  @After
  public void tearDown() {
    for (File f : processor.directory.listFiles()) {
      assertTrue("Could not delete file: " + f.getPath(), f.delete());
    }
    assertTrue("Could not delete temporary directory", processor.directory.delete());
  }

  @Test
  public void testSetUpSuccessful() {
    assertNotNull(processor);
  }

  @Test
  public void testValidCountry() {
    // Simple Cases
    assertTrue(processor.isValidCountry("united states"));
    assertTrue(processor.isValidCountry(" United States"));
    assertTrue(processor.isValidCountry("ARGENTINA"));
    assertFalse(processor.isValidCountry("Petoria"));
    // Alternate Names
    assertTrue(processor.isValidCountry("U.S."));
    assertTrue(processor.isValidCountry("US"));
    assertFalse(processor.isValidCountry("us"));
    assertTrue(processor.isValidCountry("Usa"));
    assertFalse(processor.isValidCountry("usa"));
  }

  @Test
  public void testValidRegion() {
    assertTrue(processor.isValidRegion("california"));
    assertTrue(processor.isValidRegion("California"));
    assertTrue(processor.isValidRegion("NEW YORK  "));
    assertTrue(processor.isValidRegion("CA"));
    assertFalse(processor.isValidRegion("Cali"));
  }

  @Test
  public void testValidCity() {
    assertTrue(processor.isValidCity("san francisco"));
    assertTrue(processor.isValidCity("San Francisco  "));
    assertTrue(processor.isValidCity("NEW YORK"));
    assertTrue(processor.isValidCity("SF"));
    assertFalse(processor.isValidCity("NY"));
  }

  @Test
  public void testCityRegionConsistency() {
    assertTrue(processor.consistentCityRegion("san francisco", "california"));
    assertTrue(processor.consistentCityRegion("san Francisco ", "california"));
    assertTrue(processor.consistentCityRegion("san Francisco ", "new mexico")); // there's apparently another SF here?
    assertFalse(processor.consistentCityRegion("san francisco", "ohio"));
  }

  @Test
  public void testCityCountryConsistency() {
    assertTrue(processor.consistentCityCountry("san francisco", "united states"));
    assertTrue(processor.consistentCityCountry("san Francisco ", "United States   "));
    assertTrue(processor.consistentCityCountry("san Francisco ", "Argentina"));
    assertFalse(processor.consistentCityCountry("san francisco", "England"));
  }

  @Test
  public void testRegionCountryConsistency() {
    assertTrue(processor.consistentRegionCountry("california", "united states"));
    assertTrue(processor.consistentRegionCountry("California  ", " united States"));
    assertTrue(processor.consistentRegionCountry("new york", "united states"));
    assertFalse(processor.consistentRegionCountry("new york", "argentina"));
  }

  @Test
  public void testGeographyConsistency() {
    Maybe<String> city = Maybe.Just("San Francisco");
    Maybe<String> region = Maybe.Just("California");
    Maybe<String> country = Maybe.Just("United States");
    assertTrue(processor.consistentGeography(city, region, country));
    city = Maybe.Just("New York");
    assertFalse(processor.consistentGeography(city, region, country));
    city = Maybe.Nothing();
    assertTrue(processor.consistentGeography(city, region, country));
    region = Maybe.Nothing();
    assertTrue(processor.consistentGeography(city, region, country));
    city = Maybe.Just("New York");
    assertTrue(processor.consistentGeography(city, region, country));
    region = Maybe.Just("New York");
    assertTrue(processor.consistentGeography(city, region, country));
    city = Maybe.Just("San Francisco");
    assertFalse(processor.consistentGeography(city, region, country));
  }

  @Test
  public void testSuggestRegion() {
    assertEquals("california", processor.regionForCity("san francisco").orCrash());
    assertEquals("new york", processor.regionForCity("new York ").orCrash());
  }

  @Test
  public void testSuggestCountry() {
    assertEquals("united states", processor.countryForCity("san francisco").orCrash());
    assertEquals("united states", processor.countryForCity(" New york").orCrash());
  }

  @Test
  public void testPopulation() {
    assertEquals(732072, (int) processor.cityPopulation("san francisco").orCrash());
    assertEquals(732072, (int) processor.cityPopulation("san Francisco  ").orCrash());
    assertEquals(8107916, (int) processor.cityPopulation("new york").orCrash());
  }

  @Test
  public void testNationality() {
    assertTrue(processor.consistentNationalityCountry("American", "united states"));
    assertTrue(processor.consistentNationalityCountry("American", "united States"));
    assertFalse(processor.consistentNationalityCountry("French", "united states"));
    assertTrue(processor.consistentNationalityCountry("American", "usa"));
    assertTrue(processor.consistentNationalityCountry("American", "U.S.A."));

    assertEquals("American", processor.nationalityForCountry("united states").get());
  }
}
