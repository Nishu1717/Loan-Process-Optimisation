# Business Requirements Document (BRD)
**Project Name:** Loan Process Optimisation  
**Version:** 1.0  
**Date:** 2026-01-21  
**Author:** Nisarg Patel  

---

## 1. Executive Summary
This document outlines the business requirements for optimizing the loan application and approval process for a financial institution. The current manual process suffers from inefficiencies, particularly in document validation and resource allocation, leading to extended cycle times and suboptimal customer experience. This project aims to respond to these challenges through data-driven analysis and process automation.

## 2. Problem Statement
Analysis of the 'As-Is' process has revealed two critical bottlenecks:
1.  **"Incomplete File" Loop**: A manual feedback loop for missing or incorrect documents causes significant delays (average delay identified in Phase 3).
2.  **Resource Laggards ("User_91" Bottleneck)**: A high variance in queue times among resources, with the bottom 10% of users ("Laggards") causing disproportionate delays compared to efficient users.

## 3. Project Objectives
-   **Minimize Cycle Time**: Reduce the overall "Time-to-Offer" for loan applications.
-   **Automate Triage**: Implement an automated system to handle incomplete applications instantly.
-   **Optimize Workforce**: Balance the workload dynamically among resources to prevent bottlenecks.
-   **Enhanced Reporting**: Provide granular visibility into process metrics (queue time, transition delay).

## 4. Functional Requirements

### 4.1. AI-OCR Document Validator (Phase 4)
-   **Requirement**: The system must automatically scan submitted documents upon the `A_Submitted` event.
-   **Logic**:
    -   If documents are incomplete/missing: Trigger `Send_Automated_SMS_Email` and loop back to the customer (bypassing human queue).
    -   If complete: Route to `W_Validate application` queue.
-   **Target**: Reduce the delay of the "Incomplete File" loop by **90%**.

### 4.2. Dynamic Resource Allocator (Phase 5)
-   **Requirement**: A dispatching algorithm that assigns new cases based on real-time resource availability and historical performance.
-   **Logic**:
    -   Identify "Laggard" users (Top 10% highest queue times).
    -   Redistribute incoming tasks from Laggards to "Efficient Users" (Bottom 50% queue times).
-   **Target**: Reduce the net queue time for applications.

## 5. Non-Functional Requirements
-   **Scalability**: The solutions must handle daily volumes of loan applications without performance degradation.
-   **Accuracy**: The AI-OCR system must have a high confidence threshold (>95%) before auto-rejecting documents; low-confidence results should route to a human for review.
-   **Integration**: Seamless integration with the existing BPM/CRM system event logs (`bpi_2017` dataset schema).

## 6. Proof of Concept (Phases 1-5 Analysis)
The project includes a Python-based analysis demonstrating the feasibility and impact of these requirements:
-   **Phase 1 (Data Hardening)**: Validated data integrity for analysis.
-   **Phase 2 (Bottleneck Analysis)**: Quantified the initial state metrics (Queue Time Heatmaps).
-   **Phase 3 (Transition Analysis)**: Pinpointed the "Ping-Pong" effect in validation.
-   **Phase 4 (To-Be Process)**: Modeled the theoretical time savings of automation.
-   **Phase 5 (Load Balancing)**: Simulated the queue time reduction via dynamic allocation.

## 7. Sign-off
This document serves as the foundation for the technical implementation and process redesign.

**Approver Name:** Nisarg Patel 
**Date:** 3rd Jan 2026
