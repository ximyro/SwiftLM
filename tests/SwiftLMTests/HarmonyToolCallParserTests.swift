import XCTest
@testable import MLXLMCommon

final class HarmonyToolCallParserTests: XCTestCase {
    func testInferGptOssUsesHarmonyFormat() {
        XCTAssertEqual(ToolCallFormat.infer(from: "gpt_oss"), .harmony)
    }

    func testParsesCommentaryToolCallWithoutCallTerminator() {
        let parser = HarmonyToolCallParser()
        let text = """
        <|channel|>analysis<|message|>Need file.<|end|><|start|>assistant<|channel|>commentary to=functions.read <|constrain|>json<|message|>{"filePath":"fixture/readme.txt"}
        """

        let calls = parser.parseEOS(text, tools: nil)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].function.name, "read")
        XCTAssertEqual(calls[0].function.arguments["filePath"], .string("fixture/readme.txt"))
    }

    func testParsesTemplateEncodedToolCall() {
        let parser = HarmonyToolCallParser()
        let text = """
        to=functions.bash<|channel|>commentary json<|message|>{"command":"printf OK"}<|call|>
        """

        let call = parser.parse(content: text, tools: nil)

        XCTAssertEqual(call?.function.name, "bash")
        XCTAssertEqual(call?.function.arguments["command"], .string("printf OK"))
    }
}
