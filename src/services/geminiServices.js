// const { GoogleGenerativeAI } = require("@google/generative-ai");
const {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} = require("@langchain/google-genai");
const {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
} = require("@langchain/core/prompts");
const {
  EmbeddingsFilter,
} = require("langchain/retrievers/document_compressors/embeddings_filter");
const {
  ContextualCompressionRetriever,
} = require("langchain/retrievers/contextual_compression");
const { ConversationalRetrievalQAChain } = require("langchain/chains");
const { BufferWindowMemory } = require("langchain/memory");
const { getRetriever } = require("./vectorStore");
const { useDirectoryLoader } = require("./fileloader");

let compression_retriever, llm;

const storeData = async () => {
  /* Create Training Data for Chatbot */
  // const documents = await usePuppeteer(urls);
  const documents = await useDirectoryLoader("public/data");

  const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "embedding-001",
  });

  const collectionName = "RIC-enhanced-FAQ";
  const retriever = await getRetriever(documents, embeddings, collectionName);
  llm = new ChatGoogleGenerativeAI({ modelName: "gemini-pro" });

  /* Creating Compression Retriever for Accurate Results */
  const embeddings_filter = new EmbeddingsFilter({
    embeddings,
    similarityThreshold: 0.7,
    k: 10,
  });

  compression_retriever = new ContextualCompressionRetriever({
    baseCompressor: embeddings_filter,
    baseRetriever: retriever,
  });
};

const submitQuestion = async (req, res) => {
  /* Creating Prompt */
  const system_template = `
  When user ask with other language such as Chinese, Japanese and Indonesia, then you translate first into English and Answer the question as detailed as possible from the provided context. Use the following pieces of context to answer the users question. 
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  ----------------
  {context}`;

  const messages = [
    SystemMessagePromptTemplate.fromTemplate(system_template),
    HumanMessagePromptTemplate.fromTemplate("{question}"),
  ];

  const prompt = ChatPromptTemplate.fromMessages(messages);

  /* Creating Memory Instance */
  const memory = new BufferWindowMemory({
    memoryKey: "chat_history",
    inputKey: "question",
    outputKey: "text",
    k: 3,
    returnMessages: true,
  });

  /* Creating Question Chain */
  const chain = ConversationalRetrievalQAChain.fromLLM(
    llm,
    compression_retriever,
    {
      returnSourceDocuments: true,
      memory: memory,
      verbose: true,
      qaChainOptions: {
        type: "stuff",
        prompt: prompt,
      },
    }
  );

  try {
    /* Invoking Chain for Q&A */
    const result = await chain.invoke({
      question: req.body.question,
      chat_history: memory,
    });
    const answer = await result.text;
    const sources = await result.sourceDocuments;
    res.status(200).json({ msg: answer });
  } catch (error) {
    console.error(error);
    res.status(400).json({ msg: "Something went wrong" });
  }
};

module.exports = {
  submitQuestion,
  storeData,
};

// const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
// const model = genAI.getGenerativeModel({ model: "gemini-pro" });

// const { question } = req.body;

// const result = await model.generateContent(question);
