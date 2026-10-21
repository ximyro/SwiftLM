import XCTest
import MLXLMCommon

final class HarmonyToolCallParserTests: XCTestCase {
    func testGptOssFormatIsNotInferredAsTextDialect() {
        XCTAssertNil(ToolCallFormat.infer(from: "gpt_oss"))
    }

    func testHarmonyFramesAreNotParsedAsTextToolCalls() {
        // Upstream routes Harmony through its token protocol decoder.
        let parser = ToolCallFormat.gptOSS.createParser()
        let text = """
        to=functions.bash<|channel|>commentary json<|message|>{"command":"printf OK"}<|call|>
        """
        XCTAssertNil(parser.parse(content: text, tools: nil))
    }
}
