const { Chroma } = require("@langchain/community/vectorstores/chroma");
const { ChromaClient } = require("chromadb");

async function getRetriever(documents, embeddings, collectionName) {
  const client = new ChromaClient();
  const collections = await client.listCollections();
  const vectorStoreConfig = {
    k: 15,
    searchType: "similarity",
  };
  if (!collections.every((collection) => collection.name !== collectionName)) {
    await client.deleteCollection({ name: collectionName });
    // console.log("Collection Already Exists");
    // const vector_store = await Chroma.fromExistingCollection(embeddings, {
    //   collectionName: collectionName,
    // });

    // return vector_store.asRetriever(vectorStoreConfig);
  }

  console.log("Creating New Collection");
  const vector_store = await Chroma.fromDocuments(documents, embeddings, {
    collectionName: collectionName,
  });

  return vector_store.asRetriever(vectorStoreConfig);
}

module.exports = { getRetriever };
