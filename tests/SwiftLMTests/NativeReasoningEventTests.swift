import Foundation
import HTTPTypes
import Hummingbird
import MLXLMCommon
import NIOCore
import XCTest

@testable import SwiftLM

final class NativeReasoningEventTests: XCTestCase {
    private enum Endpoint { case chat, chatStream, text, textStream }

    private final class BodySink: @unchecked Sendable {
        var data = Data()
    }

    private struct CollectingWriter: ResponseBodyWriter {
        let sink: BodySink
        mutating func write(_ buffer: ByteBuffer) async throws {
            sink.data.append(contentsOf: buffer.readableBytesView)
        }
        consuming func finish(_ trailingHeaders: HTTPFields?) async throws {}
    }

    private func respond(
        _ endpoint: Endpoint, events: [Generation], enableThinking: Bool = true,
        structuredReasoning: Bool = true, jsonMode: Bool = false
    ) async throws -> (reasoning: String, content: String) {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        for event in events { continuation.yield(event) }
        continuation.yield(
            .info(
                GenerateCompletionInfo(
                    promptTokenCount: 3, generationTokenCount: 9,
                    promptTime: 0, generationTime: 0, stopReason: .stop)))
        continuation.finish()
        let semaphore = AsyncSemaphore(limit: 1)
        await semaphore.wait()
        let slot = GenerationSlot(semaphore: semaphore)
        let response: Response
        switch endpoint {
        case .chat:
            response = try await handleChatNonStreaming(
                stream: stream, modelId: "test-model", stopSequences: ["STOP"],
                promptTokenCount: 3, enableThinking: enableThinking, thinkingPreOpened: true,
                structuredReasoning: structuredReasoning, jsonMode: jsonMode, tokenEcho: false,
                slot: slot, stats: ServerStats(), genStart: Date(), prefillStart: Date())
        case .chatStream:
            response = handleChatStreaming(
                startGeneration: { (stream, nil) }, modelId: "test-model", stopSequences: ["STOP"],
                includeUsage: false, promptTokenCount: 3, enableThinking: enableThinking,
                thinkingPreOpened: true, structuredReasoning: structuredReasoning,
                jsonMode: jsonMode, tokenEcho: false,
                slot: slot, stats: ServerStats(), genStart: Date(), prefillStart: Date(),
                emitPrefillProgress: false)
        case .text:
            response = try await handleTextNonStreaming(
                stream: stream, modelId: "test-model", stopSequences: ["STOP"],
                promptTokenCount: 3, slot: slot, stats: ServerStats(), genStart: Date())
        case .textStream:
            response = handleTextStreaming(
                startGeneration: { stream }, modelId: "test-model", stopSequences: ["STOP"],
                promptTokenCount: 3, slot: slot, stats: ServerStats(), genStart: Date(),
                emitPrefillProgress: false)
        }
        let sink = BodySink()
        try await response.body.write(CollectingWriter(sink: sink))
        switch endpoint {
        case .chat, .text:
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: sink.data) as? [String: Any])
            let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
            let choice = try XCTUnwrap(choices.first)
            let message = choice["message"] as? [String: Any]
            let usage = try XCTUnwrap(object["usage"] as? [String: Any])
            XCTAssertEqual(usage["completion_tokens"] as? Int, 9)
            return (
                message?["reasoning_content"] as? String ?? "",
                message?["content"] as? String ?? choice["text"] as? String ?? ""
            )
        case .chatStream, .textStream:
            var reasoning = ""
            var content = ""
            for raw in String(decoding: sink.data, as: UTF8.self).components(separatedBy: "\n") {
                let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                guard line.hasPrefix("data: {") else { continue }
                let object = try XCTUnwrap(
                    JSONSerialization.jsonObject(
                        with: Data(line.dropFirst(6).utf8)) as? [String: Any])
                let choice = (object["choices"] as? [[String: Any]])?.first
                let delta = choice?["delta"] as? [String: Any]
                reasoning += delta?["reasoning_content"] as? String ?? ""
                content += delta?["content"] as? String ?? choice?["text"] as? String ?? ""
            }
            return (reasoning, content)
        }
    }

    func testNativeReasoningStaysSeparateAndAnswerTagsStayLiteral() async throws {
        let answer = "The literal <think>tag</think> is an example."
        for endpoint: Endpoint in [.chat, .chatStream] {
            for structured in [true, false] {
                let result = try await respond(
                    endpoint,
                    events: [
                        .reasoning("private STOP thought"), .chunk(answer, tokenId: 1),
                    ], structuredReasoning: structured)
                XCTAssertEqual(result.reasoning, "private STOP thought")
                XCTAssertEqual(result.content, answer)
            }
        }
    }

    func testEmptyPreopenedReasoningDoesNotSwallowTheAnswer() async throws {
        for endpoint: Endpoint in [.chat, .chatStream] {
            let result = try await respond(endpoint, events: [.chunk("answer", tokenId: 1)])
            XCTAssertEqual(result.reasoning, "")
            XCTAssertEqual(result.content, "answer")
        }
    }

    func testDisabledThinkingDoesNotLeakReasoningIntoContent() async throws {
        for endpoint: Endpoint in [.chat, .chatStream] {
            let result = try await respond(
                endpoint,
                events: [
                    .reasoning("private thought"), .chunk("answer", tokenId: 1),
                ], enableThinking: false)
            XCTAssertEqual(result.reasoning, "")
            XCTAssertEqual(result.content, "answer")
        }
    }

    func testLegacyTaggedDFlashOutputStillSplitsThinking() async throws {
        for endpoint: Endpoint in [.chat, .chatStream] {
            let result = try await respond(
                endpoint,
                events: [
                    .chunk("private</think>answer", tokenId: 1)
                ], structuredReasoning: false)
            XCTAssertEqual(result.reasoning, "private")
            XCTAssertEqual(result.content, "answer")
        }
    }

    func testTextCompletionsExposeOnlyAnswerText() async throws {
        for endpoint: Endpoint in [.text, .textStream] {
            let result = try await respond(
                endpoint,
                events: [
                    .reasoning("private STOP thought"), .chunk("answer", tokenId: 1),
                ])
            XCTAssertEqual(result.reasoning, "")
            XCTAssertEqual(result.content, "answer")
        }
    }
    func testReasoningDoesNotPrematurelyFlushTheJSONAnswerBuffer() async throws {
        let events =
            Array(repeating: Generation.reasoning("private"), count: 32) + [
                .chunk("{\"wrong\":1}", tokenId: 1),
                .chunk("{\"right\":2}", tokenId: 2),
            ]
        let result = try await respond(.chatStream, events: events, jsonMode: true)
        XCTAssertEqual(result.content, "{\"right\":2}")
    }

}
