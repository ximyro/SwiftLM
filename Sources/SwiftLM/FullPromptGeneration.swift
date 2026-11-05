import MLXLMCommon

func fullPromptGenerationStream(
    iterator: consuming some TokenIteratorProtocol,
    promptTokens: [Int],
    configuration: ModelConfiguration,
    tokenizer: any Tokenizer,
    toolCallPolicy: ToolCallPolicy = .init()
) -> AsyncStream<Generation> {
    let primedInside =
        configuration.reasoningConfig.map {
            ReasoningEventEmitter.promptEndsInsideReasoning(
                promptTokens: promptTokens, config: $0, tokenizer: tokenizer)
        } ?? false
    return MLXLMCommon.generateTask(
        promptTokenCount: promptTokens.count,
        modelConfiguration: configuration, tokenizer: tokenizer, iterator: iterator,
        toolCallPolicy: toolCallPolicy, reasoningPrimedInside: primedInside
    ).0
}

func generateFromFullPrompt(
    input: LMInput, cache: [KVCache], parameters: GenerateParameters,
    context: ModelContext, promptTokens: [Int],
    components: GenerationComponents = .init(), numMTPTokens: Int? = nil
) throws -> AsyncStream<Generation> {
    let iterator: any TokenIteratorProtocol
    if let numMTPTokens, let model = context.model as? any MTPLanguageModel {
        iterator = try MTPTokenIterator(
            input: input, model: model, cache: cache,
            parameters: parameters, numMTPTokens: numMTPTokens)
    } else {
        iterator = try TokenIterator(
            input: input, model: context.model, cache: cache,
            parameters: parameters, components: components)
    }
    return fullPromptGenerationStream(
        iterator: iterator, promptTokens: promptTokens,
        configuration: context.configuration, tokenizer: context.tokenizer,
        toolCallPolicy: parameters.toolCallPolicy)
}
