import Foundation
import XCTest
@testable import SwiftLM

final class ReasoningEffortTests: XCTestCase {
    private func decode(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    func testMediumEffortReachesTemplateAlongsideThinkingBudget() throws {
        let request = try decode(#"{"messages":[],"reasoning_effort":"medium","thinking_budget":16384}"#)
        let context = request.templateContext(enableThinking: true)
        XCTAssertEqual(context["reasoning_effort"] as? String, "medium")
        XCTAssertEqual(context["enable_thinking"] as? Bool, true)
        XCTAssertEqual(request.thinkingBudget, 16384)
    }

    func testOmittedAndNullEffortPreserveModelTemplateDefault() throws {
        for json in [#"{"messages":[]}"#, #"{"messages":[],"reasoning_effort":null}"#] {
            let context = try decode(json).templateContext(enableThinking: true)
            XCTAssertNil(context["reasoning_effort"])
            XCTAssertEqual(context["enable_thinking"] as? Bool, true)
        }
    }

    func testEffortDoesNotOverrideResolvedThinkingSwitch() throws {
        let request = try decode(#"{"messages":[],"reasoning_effort":"medium","enable_thinking":true}"#)
        XCTAssertEqual(request.templateContext(enableThinking: false)["enable_thinking"] as? Bool, false)
    }

    func testNonStringEffortIsAClientError() throws {
        do {
            _ = try decode(#"{"messages":[],"reasoning_effort":42}"#)
            XCTFail("numeric effort must be rejected")
        } catch let error as DecodingError {
            let body = invalidRequestJSON(error)
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
            let detail = try XCTUnwrap(json["error"] as? [String: Any])
            XCTAssertEqual(detail["param"] as? String, "reasoning_effort")
            XCTAssertEqual(detail["type"] as? String, "invalid_request_error")
        }
    }
}
