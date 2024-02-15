const { DirectoryLoader } = require("langchain/document_loaders/fs/directory");
const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
const { splitDocuments } = require("./splitDocuments");

async function useDirectoryLoader(directory) {
  /* Load all PDFs within the specified directory */
  try {
    const directoryLoader = new DirectoryLoader(directory, {
      ".pdf": (path) => new PDFLoader(path),
    });

    const docs = await directoryLoader.load();

    const { documents } = await splitDocuments(docs);

    return documents;
  } catch (error) {
    console.error(error);
  }
}

module.exports = { useDirectoryLoader };
