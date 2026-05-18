import XCTest
@testable import SwiftLM
import MLXLMCommon

final class Gemma4ToolSchemaTests: XCTestCase {
    func testGemma4NormalizesNullableTypeArray() throws {
        let request = try decodeRequest(
            parameters: """
            {
              "type": "object",
              "properties": {
                "path": {
                  "type": ["string", "null"],
                  "description": "File path"
                }
              },
              "required": ["path"]
            }
            """
        )

        let property = try propertySchema(from: request, format: .gemma4, name: "path")
        XCTAssertEqual(property["type"] as? String, "string")
        XCTAssertEqual(property["nullable"] as? Bool, true)
        XCTAssertEqual(property["description"] as? String, "File path")
    }

    func testGemma4FlattensNullableAnyOf() throws {
        let request = try decodeRequest(
            parameters: """
            {
              "type": "object",
              "properties": {
                "query": {
                  "anyOf": [
                    { "type": "string" },
                    { "type": "null" }
                  ]
                }
              }
            }
            """
        )

        let property = try propertySchema(from: request, format: .gemma4, name: "query")
        XCTAssertEqual(property["type"] as? String, "string")
        XCTAssertNil(property["anyOf"])
    }

    func testGemma4NormalizesNestedObjectsAndArrays() throws {
        let request = try decodeRequest(
            parameters: """
            {
              "type": "object",
              "properties": {
                "settings": {
                  "type": "object",
                  "properties": {
                    "tags": {
                      "type": "array",
                      "items": {
                        "anyOf": [
                          { "type": "string" },
                          { "type": "null" }
                        ]
                      }
                    }
                  },
                  "required": ["tags"]
                }
              }
            }
            """
        )

        let settings = try propertySchema(from: request, format: .gemma4, name: "settings")
        XCTAssertEqual(settings["type"] as? String, "object")

        let nestedProperties = try XCTUnwrap(settings["properties"] as? [String: any Sendable])
        let tags = try XCTUnwrap(nestedProperties["tags"] as? [String: any Sendable])
        XCTAssertEqual(tags["type"] as? String, "array")

        let items = try XCTUnwrap(tags["items"] as? [String: any Sendable])
        XCTAssertEqual(items["type"] as? String, "string")
        XCTAssertNil(items["anyOf"])
    }

    func testNonGemmaKeepsOriginalSchemaShape() throws {
        let request = try decodeRequest(
            parameters: """
            {
              "type": "object",
              "properties": {
                "path": {
                  "type": ["string", "null"]
                }
              }
            }
            """
        )

        let property = try propertySchema(from: request, format: .json, name: "path")
        let type = try XCTUnwrap(property["type"] as? [any Sendable])
        XCTAssertEqual(type[0] as? String, "string")
        XCTAssertEqual(type[1] as? String, "null")
        XCTAssertNil(property["nullable"])
    }

    private func decodeRequest(parameters: String) throws -> ChatCompletionRequest {
        let json = """
        {
          "model": "test",
          "messages": [{ "role": "user", "content": "hello" }],
          "tools": [
            {
              "type": "function",
              "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": \(parameters)
              }
            }
          ]
        }
        """
        let data = try XCTUnwrap(json.data(using: .utf8))
        return try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
    }

    private func propertySchema(
        from request: ChatCompletionRequest,
        format: ToolCallFormat,
        name: String
    ) throws -> [String: any Sendable] {
        let tools = try XCTUnwrap(makeTemplateToolSpecs(request.tools, toolCallFormat: format))
        let function = try XCTUnwrap(tools[0]["function"] as? [String: any Sendable])
        let parameters = try XCTUnwrap(function["parameters"] as? [String: any Sendable])
        let properties = try XCTUnwrap(parameters["properties"] as? [String: any Sendable])
        return try XCTUnwrap(properties[name] as? [String: any Sendable])
    }
}
