const { GoogleGenerativeAI } = require("@google/generative-ai");
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
  // const system_template = `
  // When user ask with other language such as Chinese, Japanese and Indonesia, then you translate first into English and Answer the question as detailed as possible from the provided context. Use the following pieces of context to answer the users question.
  // If you don't know the answer, just say that you don't know, don't try to make up an answer.
  // ----------------
  // {context}`;
  // const messages = [
  //   SystemMessagePromptTemplate.fromTemplate(system_template),
  //   HumanMessagePromptTemplate.fromTemplate("{question}"),
  // ];
  // const prompt = ChatPromptTemplate.fromMessages(messages);
  // /* Creating Memory Instance */
  // const memory = new BufferWindowMemory({
  //   memoryKey: "chat_history",
  //   inputKey: "question",
  //   outputKey: "text",
  //   k: 3,
  //   returnMessages: true,
  // });
  // /* Creating Question Chain */
  // const chain = ConversationalRetrievalQAChain.fromLLM(
  //   llm,
  //   compression_retriever,
  //   {
  //     returnSourceDocuments: true,
  //     memory: memory,
  //     verbose: true,
  //     qaChainOptions: {
  //       type: "stuff",
  //       prompt: prompt,
  //     },
  //   }
  // );
  // try {
  //   /* Invoking Chain for Q&A */
  //   const result = await chain.invoke({
  //     question: req.body.question,
  //     chat_history: memory,
  //   });
  //   const answer = await result.text;
  //   const sources = await result.sourceDocuments;
  //   res.status(200).json({ msg: answer });
  // } catch (error) {
  //   console.error(error);
  //   res.status(400).json({ msg: "Something went wrong" });
  // }
  try {
    const genAI = new GoogleGenerativeAI(
      "AIzaSyCBFARQS8sOgOK0Klc__Gtdsei8plZpWfQ"
    );
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });
    const chat = model.startChat({
      history: [
        {
          role: "user",
          parts:
            "how many options for internship at ICT Program Rangsit International college?",
        },
        {
          role: "model",
          parts:
            "There are two options for Internship at RIC ICT Program (Option 1) Internship and Senior Project (6 credits) and (Option 2) ICT 498 Co-operative Education (6 credits)",
        },
        {
          role: "user",
          parts: "Tell me details of internship options.",
        },
        {
          role: "model",
          parts:
            "There are two options for Internship at RIC ICT Program (Option 1) Internship and Senior Project (6 credits) 1(0-35-18) ICT 493 Internship (1 credit) Students will enter an industrial training program with the industrial sector, the business sector, or other public organizations for a period of at least 280 hours. Students need to work at least 280 hours or 2 months for internship. Student is required to do the report. 2(2-0-4) ICT 494 Introduction to Senior Project (2 credits) Before you start the internship or co operative education, Students must earned at least 45 credits of required major courses or acquire consent from Head of Program. Creation of senior project topics; outlining and drafting senior projects; citations and reference formats; writing introduction for senior projects; compilation of theories; frameworks and existed works and/or related researches; writing for senior project methodologies; development and presentation of models and/or tools used in senior projects; preliminary examination of senior project proposals. 3(0-6-3) ICT 495 Senior Project (3 credits) . Students must finished ICT 494 Introduction to Senior Project and/or must earned at least 50 credits of required major courses or acquire consent of Head of Program. Completion of senior projects; printing standard for full senior projects; preparation for final examination of senior projects; final examination of senior projects. 6(0-35-18). (Option 2) ICT 498 Co-operative Education (6 credits) Students must earned at least 60 credits of major courses or acquire consent from Head of Program. Minimum of 16 weeks or 560 hours must be achieved to complete co operative education. This program is designed to enhance traditional academic study by providing students the opportunity to integrate academic study and classroom theory with productive and real-life work experience in Information and Communication Technology field at a work site in the community. It also provides the time to explore occupational areas and to develop skills, knowledge and attitudes needed to become productive and satisfied members of society.",
        },
        {
          role: "user",
          parts: "Timeline for internship and co-operative",
        },
        {
          role: "model",
          parts:
            "There are 12 steps for Timeline for internship and co-operative, each step has completion percentages:\n1. Looking for work placement (ICTCE01 Co-Operative Education Application Form) - 0% complete.\n2. Work Placement Accept - 5% complete.\n3. Job Description form sign-up from workplace (ICTCE02 Co-Operative Education Job Description form) - 10% complete.\n4. ICT Department issue the letter for work placement (Letter of request for work placement * giving to the employer) - 10% complete.\n5. Teacher visiting (ICTCE03 confirmation form of workplace supervision of Coop Edu supervision program) - 15% complete.\n6. Weekly progress report (ICTCE04 Weekly process report form) - 35% complete.\n7. Evaluate work performance (ICTCE05 Co-operative education evaluation form from job supervisor) - 50% complete.\n8. Evaluate internship report (ICTCE06 Student evaluation form from job advisor/supervisor) - 60% complete.\n9. Consent to publish report (ICTCE07 – Letter of consent to publish report/co-operative education project) - 70% complete.\n10. Submit Report (cooperative education report for 1 job supervisor, 2 co-operative supervisors, 3 the committee) - 80% complete.\n11. Presentation (Student’s evaluation form (for job advisor/supervisor for 1 job supervisor, 2 co-operative supervisors, 3 the committee) - 90% complete.\n12. Amendment & Submit final draft report (within 2 weeks after presentation) - 100% complete.",
        },
        {
          role: "user",
          parts:
            "Tell me steps of GRADUATION REQUEST PROCESS RANGSIT UNIVERSITY",
        },
        {
          role: "model",
          parts:
            "Keeping in mind to note the specific period for near-graduated students to request graduation according to the academic calendar, the steps are:\nStart Graduation Request (0%).\nVisit the Registrar’s office website or directly access http://registrar.rsu.ac.th/applyforgraduate (period for near-graduated).\nAuthentication (10%).\nLog in with your Username (with 'u' in front) and Password (same as the Intranet system).\nInput Information (20%).\nFill in your address details for document dispatch.\nPrint Forms (30%).\nPrint out the Curriculum Structure Form (RSU.36) and Graduation Request Form (RSU.62).\nPayment (40%).\nPay the 200 Baht fee via the provided payment channels (QR code available).\nPhoto Submission (50%).\nObtain a 2-inch photo in graduation gown against a blue background.\nAdvisor Approval (60%).\nGet the Curriculum Structure Form (RSU.36) approved and signed by your advisor.\nForm Submission (70%).\nSubmit the RSU.36 and RSU.62 forms along with the photograph to the Registrar’s office before the deadline (period for near-graduated).\nConfirmation (80%).\nReceive acknowledgment of submission from the Registrar’s office.\nFinal Verification (90%).\nWait for the verification process by the University Council.\nGraduation Approval (100%).\nReceive confirmation of graduation approval.",
        },
        {
          role: "user",
          parts:
            "How many forms do I need to fill for internship and co-operative education?",
        },
        {
          role: "model",
          parts:
            "There are 7 application forms for internship and co-operative education:\n1. ICTCE01 - Looking for work placement (Co-operative Education Application Form)\n2. ICTCE02 - Job description form sign-up from workplace (Co-operative Education Job Description Form)\n3. ICTCE03 - (Confirmation Form of Workplace Supervision of Co-operative Education Supervision Program)\n4. ICTCE04 - Weekly Progress Report (Weekly Progress Report Form)\n5. ICTCE05 - Evaluate Work Performance Co-operative Education Evaluation Form (For Job Supervisor)\n6. ICTCE06 - Evaluate Internship Report Student’s Evaluation Form (For Job Advisor/Supervisor)\n7. ICTCE07 - Consent to Publish Report (Letter of Consent to Publish Report/Co-operative Education Project)",
        },
        {
          role: "user",
          parts: "How many internship options are there in RIC?",
        },
        {
          role: "model",
          parts:
            "There are two internship options: 1. Internship and 2. Co-operative Education.",
        },
        {
          role: "user",
          parts: "How to find internship or co-operative education?",
        },
        {
          role: "model",
          parts:
            "You can try to find internships from these platforms: glints, internjobs.com, experience by simplicity, internsg, glassdoor, hiredly, LinkedIn, wayup, Asia Internship program, and gradsingapore.",
        },
        {
          role: "user",
          parts: "What are the admissions requirements?",
        },
        {
          role: "model",
          parts:
            "The academic entry requirement is a high school diploma or equivalent and IELTS 5 or equivalent.",
        },
        {
          role: "user",
          parts: "What if I don’t have any recent IELTS scores?",
        },
        {
          role: "model",
          parts:
            "Students can take an English Placement Test (free-of-charge) once they arrive at the university. If a student’s English level is below the minimum requirement of IELTS 5, the student would need to take some English subjects before starting the program. Please see the section English Placement Test for more details.",
        },
        {
          role: "user",
          parts: "How long does the admissions process take?",
        },
        {
          role: "model",
          parts:
            "Once we have the completed application file, including all required materials, it usually takes a few days to a week for application processing, review, and for your Letter of Acceptance to be sent by email. You will then be required to pay a non-refundable partial payment 30% of tuition fees of the semester before your visa supporting document is granted. The remaining 70% of the fees are paid once you arrived at the university. Please note, the partial payment may be changed based on your country of origin.",
        },
        {
          role: "user",
          parts: "Where do I get my visa and how long does it take to have it?",
        },
        {
          role: "user",
          parts:
            "Tell me steps of GRADUATION REQUEST PROCESS RANGSIT UNIVERSITY",
        },
        {
          role: "model",
          parts:
            "Keeping in mind to note the specific period for near-graduated students to request graduation according to the academic calendar, the steps are:\nStart Graduation Request (0%).\nVisit the Registrar’s office website or directly access http://registrar.rsu.ac.th/applyforgraduate (period for near-graduated).\nAuthentication (10%).\nLog in with your Username (with 'u' in front) and Password (same as the Intranet system).\nInput Information (20%).\nFill in your address details for document dispatch.\nPrint Forms (30%).\nPrint out the Curriculum Structure Form (RSU.36) and Graduation Request Form (RSU.62).\nPayment (40%).\nPay the 200 Baht fee via the provided payment channels (QR code available).\nPhoto Submission (50%).\nObtain a 2-inch photo in graduation gown against a blue background.\nAdvisor Approval (60%).\nGet the Curriculum Structure Form (RSU.36) approved and signed by your advisor.\nForm Submission (70%).\nSubmit the RSU.36 and RSU.62 forms along with the photograph to the Registrar’s office before the deadline (period for near-graduated).\nConfirmation (80%).\nReceive acknowledgment of submission from the Registrar’s office.\nFinal Verification (90%).\nWait for the verification process by the University Council.\nGraduation Approval (100%).\nReceive confirmation of graduation approval.",
        },
        {
          role: "user",
          parts:
            "How many forms do I need to fill for internship and co-operative education?",
        },
        {
          role: "model",
          parts:
            "There are 7 application forms for internship and co-operative education:\n1. ICTCE01 - Looking for work placement (Co-operative Education Application Form)\n2. ICTCE02 - Job description form sign up from workplace (Co-operative Education Job Description Form)\n3. ICTCE03 - (Confirmation Form of Workplace Supervision of Co-operative Education Supervision Program)\n4. ICTCE04 - Weekly Progress Report (Weekly Progress Report Form)\n5. ICTCE05 - Evaluate Work Performance Co-operative Education Evaluation Form (For Job Supervisor)\n6. ICTCE06 - Evaluate Internship Report Student’s Evaluation Form (For Job Advisor/Supervisor)\n7. ICTCE07 - Consent to Publish Report (Letter of Consent to Publish Report/Co-operative Education Project)",
        },
        {
          role: "user",
          parts: "How many internship options are there in RIC?",
        },
        {
          role: "model",
          parts:
            "There are two internship options: 1. Internship and 2. Co-operative Education.",
        },
        {
          role: "user",
          parts: "How to find internship or co-operative education?",
        },
        {
          role: "model",
          parts:
            "You can try to find internships from these platforms: glints, internjobs.com, experience by simplicity, internsg, glassdoor, hiredly, LinkedIn, wayup, Asia Internship program, and gradsingapore.",
        },
      ],
    });
    const { question } = req.body;
    const result = await model.generateContent(question);
    const response = await result.response;
    const text = response.text();
    res.status(200).json({ msg: text });
  } catch (error) {
    console.error(error);
    res.status(400).json({ msg: "Something went wrong" });
  }
};

module.exports = {
  submitQuestion,
  storeData,
};
