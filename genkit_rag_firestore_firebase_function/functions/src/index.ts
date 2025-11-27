import {genkit, z} from "genkit";
import {vertexAI} from "@genkit-ai/vertexai";
import path from 'path';
import {readFile} from "fs/promises";
import {PDFParse} from 'pdf-parse';
import {chunk} from 'llm-chunk';
import {Document} from 'genkit/retriever';
import {enableFirebaseTelemetry} from "@genkit-ai/firebase";
import {initializeApp} from "firebase-admin/app";
import {getFirestore, DocumentData, FieldValue} from 'firebase-admin/firestore';
import {defineFirestoreRetriever} from '@genkit-ai/firebase';


// DEV only
import dotenv from 'dotenv'

dotenv.config({
    path: path.resolve(__dirname, '../../.env')
});


const app = initializeApp();
let firestore = getFirestore(app);

// if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
//     const dataBuffer = await readFile(process.env.GOOGLE_APPLICATION_CREDENTIALS);
//     const serviceAccountCreds = JSON.parse(process.env.GOOGLE_APPLICATION_CREDENTIALS);
//     console.log(serviceAccountCreds);
//     const authOptions = { credentials: serviceAccountCreds };
//     firestore.settings(authOptions);
// }


// import { defineSecret } from "firebase-functions/params";
// const apiKey = defineSecret("GOOGLE_GENAI_API_KEY");


// const apiKey = defineSecret("GEMINI_API_KEY");


const location = process.env.GCLOUD_LOCATION || '';


enableFirebaseTelemetry();


const collectionName = "hr_documents";

const ai = genkit({
    plugins: [
        vertexAI({location: location,}),
    ],

});


const ragFlow = ai.defineFlow({
        name: "ragFlow",
        outputSchema: z.string(),
    }, async (subject, {sendChunk}) => {


        // Load file content
        const pdfText = await ai.run("extraction", async (): Promise<string> => {
            const pdfFile = path.resolve('./assets/HR-Policies-Manuals.pdf');
            const dataBuffer = await readFile(pdfFile);
            const pdfData = new PDFParse({data: dataBuffer});
            const textResult = await pdfData.getText() || '';
            console.log('PDF text extracted, length: ' + textResult.text.length);
            return textResult.text;
        });

        console.log('===> File loaded Done');


        // Chunking using Split method and create a Document List
        const documentList: Document[] = await ai.run("chunking", async () => {

            const chunkedData = chunk(pdfText, {
                maxLength: 5000,
                minLength: 4000,
                splitter: 'sentence',
                overlap: 100,
                delimiters: '',
            });
            console.log('Chunked content length ', chunkedData.length);

            let documents = [];

            for (let i = 0; i < chunkedData.length; i++) {
                console.log(i);

                const embedding = (await ai.embed({
                    embedder: vertexAI.embedder('text-embedding-005'),
                    content: chunkedData[i],
                }))[0].embedding;

                await new Promise((resolve) => setTimeout(resolve, 10000));


                const document: DocumentData = {
                    content: chunkedData[i],
                    embedding: FieldValue.vector(embedding),
                    timeStamp: FieldValue.serverTimestamp()
                };
                documents.push(document);

            }


            console.log('Documents  length ', documents.length);

            return documents;
        })

        // Update Database
        const batch = firestore.batch();
        documentList.forEach((doc) => {
            const ref = firestore.collection(collectionName).doc();
            batch.create(ref, doc);
        });
        await batch.commit();

        console.log('===> Database Updates');
        console.log('===> Indexing Done');


        return "Indexing Done";
    }
);


const retrieveResult = ai.defineFlow({
    name: "retrieveResult",
    inputSchema: z.string(),
    outputSchema: z.string(),
}, async input => {

    const firestoreQueryRetriever = defineFirestoreRetriever(ai, {
        name: 'firestoreQueryRetriever',
        firestore,
        collection: collectionName,
        contentField: 'content', // Field containing document content
        vectorField: 'embedding', // Field containing vector embeddings
        embedder: vertexAI.embedder('text-embedding-005'), // Embedder to generate embeddings
        distanceMeasure: 'COSINE', // Default is 'COSINE'; other options: 'EUCLIDEAN', 'DOT_PRODUCT'
    });

    const docs = await ai.retrieve({
        retriever:firestoreQueryRetriever,
        query: input.toString(),
        options: {
            limit: 2,
        },
    });

    let resultAsText="";

    for (let i = 0; i < docs.length; i++) {
        resultAsText+=`Page ${i+1}: ${docs[i].text} \n\n `
    }

    console.log(resultAsText);


    const prompt =
        `Be a helpful assistant and Answer user query in simple as informative , short and only relevant answer, \n User Query: ${input} \n Context ${resultAsText}`;
    const { response  } = ai.generateStream({
        model: vertexAI.model('gemini-2.5-pro'),
        prompt: prompt,
        config: {
            temperature: 1,
        },
    });



    return (await response).text;


});

// export const menuSuggestion = onCallGenkit({
//   secrets: [apiKey],
// }, menuSuggestionFlow);
