import MLXLMCommon
import XCTest

@testable import SwiftLM

final class CachedReasoningGenerationTests: XCTestCase {
    private struct Iterator: TokenIteratorProtocol {
        let tokens: [Int]
        var tokenCount = 0
        var maxTokens: Int? { tokens.count + 1 }
        var promptPrefillTime: Double { 0 }
        var streamingError: SSDStreamingError? { nil }
        var acceptedDraftTokens: Int { 0 }
        var totalDraftTokens: Int { 0 }

        mutating func next() -> Int? {
            guard tokenCount < tokens.count else { return nil }
            defer { tokenCount += 1 }
            return tokens[tokenCount]
        }
    }

    private struct ByteTokenizer: MLXLMCommon.Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.compactMap(UInt8.init(exactly:)), as: UTF8.self)
        }
        func convertTokenToId(_ token: String) -> Int? {
            token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
        }
        func convertIdToToken(_ id: Int) -> String? {
            UInt8(exactly: id).map { String(decoding: [$0], as: UTF8.self) }
        }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    func testCachedSuffixDoesNotLoseTheFullPromptsReasoningState() async {
        let tokenizer = ByteTokenizer()
        let reasoning = ReasoningConfig.thinkTagsWithEnableThinking
        let prompt = tokenizer.encode(text: "user\nassistant\n<think>\n", addSpecialTokens: false)
        XCTAssertFalse(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                promptTokens: Array(prompt.suffix(1)), config: reasoning, tokenizer: tokenizer))
        var configuration = ModelConfiguration(id: "cached-reasoning-test")
        configuration.reasoningConfig = reasoning
        let stream = fullPromptGenerationStream(
            iterator: Iterator(
                tokens: tokenizer.encode(
                    text: "private</think>answer", addSpecialTokens: false)),
            promptTokens: prompt, configuration: configuration, tokenizer: tokenizer)
        var thought = ""
        var answer = ""
        for await event in stream {
            thought += event.reasoning ?? ""
            answer += event.chunk ?? ""
        }
        XCTAssertEqual(thought, "private")
        XCTAssertEqual(answer, "answer")
    }
}
