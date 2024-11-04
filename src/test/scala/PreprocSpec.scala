import Preprocessor.wordTokenize
import org.scalatest.flatspec._
import org.scalatest.matchers.should._

class PreprocSpec extends AnyFlatSpec with Matchers {

  "The sentence tokenizer" should "split simple text correctly" in {
    val text1 = "This is CS 441."
    val sents1 = Preprocessor.sentTokenize(text1)
    sents1 should contain theSameElementsInOrderAs List("This is CS 441.")

    val text2 = "This is the first sentence. This is the second sentence."
    val sents2 = Preprocessor.sentTokenize(text2)
    sents2 should contain theSameElementsInOrderAs List("This is the first sentence.", "This is the second sentence.")

    val text3 = "Hi! Cutie boy"
    val sents3 = Preprocessor.sentTokenize(text3)
    sents3 should contain theSameElementsInOrderAs List("Hi!", "Cutie boy")
  }

  it should "split complex text correctly" in {
    val text1 = "Wow! Is this the first sentence? Huh?!?! I didn't know that."
    val sents1 = Preprocessor.sentTokenize(text1)
    sents1 should contain theSameElementsInOrderAs List("Wow!", "Is this the first sentence?", "Huh?!?!", "I didn't know that.")

    val text2 = "Hyphen-emdash---semicolon;comma,end."
    val sents2 = Preprocessor.sentTokenize(text2)
    sents2 should contain theSameElementsInOrderAs List("Hyphen-emdash---semicolon;comma,end.")

    val text3 = "Time ! flies @ like # an $ arrow % fruit ^ flies & like * a ( banana )"
    val sents3 = Preprocessor.sentTokenize(text3)
    sents3 should contain theSameElementsInOrderAs List("Time ! flies @ like # an $ arrow % fruit ^ flies & like * a ( banana )")
  }

  "The word tokenizer" should "tokenize simple text correctly" in {
    val text1 = "This is CS 441."
    val words1 = Preprocessor.wordTokenize(text1)
    words1 should contain theSameElementsInOrderAs List("This", "is", "CS", "441")

    val text2 = "This is the first sentence. This is the second sentence."
    val words2 = Preprocessor.wordTokenize(text2)
    words2 should contain theSameElementsInOrderAs List("This", "is", "the", "first", "sentence", "This", "is", "the", "second", "sentence")

    val text3 = "Hi! Cutie boy"
    val words3 = Preprocessor.wordTokenize(text3)
    words3 should contain theSameElementsInOrderAs List("Hi", "Cutie", "boy")
  }

  it should "split complex text correctly" in {
    val text1 = "Wow! Is this the first sentence? Huh?!?! I didn't know that."
    val words1 = Preprocessor.wordTokenize(text1)
    words1 should contain theSameElementsInOrderAs List("Wow", "Is", "this", "the", "first", "sentence", "Huh", "I", "didn", "t", "know", "that")

    val text2 = "Hyphen-emdash---semicolon;comma,end."
    val words2 = Preprocessor.wordTokenize(text2)
    words2 should contain theSameElementsInOrderAs List("Hyphen", "emdash", "semicolon", "comma", "end")

    val text3 = "Time ! flies @ like # an $ arrow % fruit ^ flies & like * a ( banana )"
    val words3 = Preprocessor.wordTokenize(text3)
    words3 should contain theSameElementsInOrderAs List("Time", "flies", "like", "an", "arrow", "fruit", "flies", "like", "a", "banana")
  }

  "Vector embeddings" should "have the correct shape" in {
    val text = "Time ! flies @ like # an $ arrow % fruit ^ flies & like * a ( banana )"
    val embeddings = Preprocessor.encodeAndEmbed(wordTokenize(text))
    embeddings.shape()(0) should equal (10)
    embeddings.shape()(1) should equal (Constants.EMBEDDING_DIM)
  }

  "Positional embeddings" should "have the correct shape" in {
    val pe = Preprocessor.positionalEmbedding
    pe.shape()(0) should equal (Constants.WINDOW_SIZE)
    pe.shape()(1) should equal (Constants.EMBEDDING_DIM)
  }
}
