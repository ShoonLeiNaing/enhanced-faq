const {
  HtmlToTextTransformer,
} = require("@langchain/community/document_transformers/html_to_text");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const fs = require("node:fs");
const path = require("path");

const filePath = path.join(__dirname, "../logs/dataloaderLog.txt");

async function splitDocuments(docs) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1024,
    chunkOverlap: 20,
  });
  const transformer = new HtmlToTextTransformer();
  const sequence = splitter.pipe(transformer);
  const documents = await sequence.invoke(docs.flat());

  /* Getting Logs */
  fs.writeFile(filePath, "", (err) => {
    if (err) console.log(err);
  });

  documents.flat().forEach((d, i) => {
    fs.appendFile(
      filePath,
      `\n${i}\n${d.metadata.source}\n${d.pageContent}\n`,
      (err) => {
        if (err) console.log(err);
      }
    );
  });

  return { documents };
}

module.exports = {
  splitDocuments,
};
